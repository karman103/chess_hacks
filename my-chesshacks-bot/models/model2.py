import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import chess
import chess.engine
import chess.pgn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
####################################################
# EXTRA: EVALUATION VS ALL STOCKFISH LEVELS (0–20)
####################################################

def model_move(board, model, sims=32, device="cpu"):
    mcts = MCTS(model, sims=sims, cpuct=1.5, device=device)
    pi = mcts.run(board)
    idx = int(torch.argmax(pi))
    move = index_to_move(idx)
    if move not in board.legal_moves:
        move = random.choice(list(board.legal_moves))
    return move


def play_game_vs_sf(model, sf_path, sf_level, model_white, sims, device):
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(sf_path)

    # Stockfish config per level
    sf_config = {
        "Skill Level": sf_level,
        "UCI_LimitStrength": True,
        "UCI_Elo": 1350 + sf_level * 75  # approximate scaling
    }
    engine.configure(sf_config)

    while not board.is_game_over():
        if (board.turn == chess.WHITE and model_white) or \
           (board.turn == chess.BLACK and not model_white):
            move = model_move(board, model, sims=sims, device=device)
        else:
            result = engine.play(board, chess.engine.Limit(time=0.05))
            move = result.move
        board.push(move)

    engine.quit()
    return board.result()


def elo_from_results(results):
    wins = results["1-0"]
    losses = results["0-1"]
    draws = results["1/2-1/2"]

    N = wins + losses + draws
    if N == 0:
        return 0

    score = (wins + 0.5 * draws) / N
    score = max(1e-6, min(1 - 1e-6, score))

    return -400 * math.log10((1 / score) - 1)


def eval_all_levels(model_path, sf_path, games, sims, device):
    # Load model
    model = PolicyValueNet().to(device)
    raw = torch.load(model_path, map_location=device)
    fixed = {k.replace("module.", ""): v for k, v in raw.items()}
    model.load_state_dict(fixed)
    model.eval()

    SKILL_LEVELS = list(range(0, 21))
    summary = {}

    for lvl in SKILL_LEVELS:
        print(f"\n======================")
        print(f" Evaluating Stockfish Level {lvl}")
        print(f"======================")

        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}

        for g in range(games):
            model_white = (g % 2 == 0)
            res = play_game_vs_sf(
                model=model,
                sf_path=sf_path,
                sf_level=lvl,
                model_white=model_white,
                sims=sims,
                device=device,
            )
            results[res] += 1
            print(f"  Game {g+1}/{games}: {res}")

        dElo = elo_from_results(results)
        sf_elo = 1150 + 75 * (lvl - 1)
        model_elo = sf_elo + dElo

        summary[lvl] = {
            "results": results,
            "elo_diff": dElo,
            "approx_model_elo": model_elo,
        }

        print(f"Results vs Level {lvl}: {results}")
        print(f"Elo diff: {dElo:.1f}")
        print(f"Model Elo: {model_elo:.1f}")

    # Final table
    print("\n\n======================")
    print(" FINAL SUMMARY TABLE")
    print("======================\n")

    print(f"{'Lvl':<5}{'W':<4}{'L':<4}{'D':<4}{'ΔElo':<10}{'ModelElo'}")
    print("-" * 45)

    for lvl in SKILL_LEVELS:
        r = summary[lvl]["results"]
        de = summary[lvl]["elo_diff"]
        me = summary[lvl]["approx_model_elo"]
        print(f"{lvl:<5}{r['1-0']:<4}{r['0-1']:<4}{r['1/2-1/2']:<4}{de:<10.1f}{me:.1f}")


# ===============================
# Device selection
# ===============================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ===============================
# Encodings and move indexing
# ===============================

PIECE_TO_PLANE = {
    (chess.PAWN, True): 0,
    (chess.KNIGHT, True): 1,
    (chess.BISHOP, True): 2,
    (chess.ROOK, True): 3,
    (chess.QUEEN, True): 4,
    (chess.KING, True): 5,
    (chess.PAWN, False): 6,
    (chess.KNIGHT, False): 7,
    (chess.BISHOP, False): 8,
    (chess.ROOK, False): 9,
    (chess.QUEEN, False): 10,
    (chess.KING, False): 11,
}


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Encode a chess.Board into a (17, 8, 8) float32 tensor.
    12 planes: piece type / color
    1 plane: side to move
    4 planes: castling rights
    """
    import numpy as np

    planes = np.zeros((17, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        plane = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
        r = 7 - chess.square_rank(square)
        c = chess.square_file(square)
        planes[plane, r, c] = 1.0

    # side to move
    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    # castling rights
    planes[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    return torch.from_numpy(planes)


def move_to_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


def index_to_move(idx: int) -> chess.Move:
    from_sq = idx // 64
    to_sq = idx % 64
    return chess.Move(from_sq, to_sq)


# ===============================
# Policy + Value Network
# ===============================


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class PolicyValueNet(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 6):
        super().__init__()
        self.stem = nn.Conv2d(17, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)

        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        # policy head
        self.conv_policy = nn.Conv2d(channels, 32, 1)
        self.bn_policy = nn.BatchNorm2d(32)
        self.fc_policy = nn.Linear(32 * 8 * 8, 64 * 64)  # 4096 moves

        # value head
        self.conv_value = nn.Conv2d(channels, 32, 1)
        self.bn_value = nn.BatchNorm2d(32)
        self.fc_value1 = nn.Linear(32 * 8 * 8, 128)
        self.fc_value2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, 17, 8, 8)
        x = F.relu(self.stem_bn(self.stem(x)))
        x = self.blocks(x)

        # policy
        p = F.relu(self.bn_policy(self.conv_policy(x)))
        p = p.view(p.size(0), -1)
        policy_logits = self.fc_policy(p)  # (B, 4096)

        # value
        v = F.relu(self.bn_value(self.conv_value(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.fc_value1(v))
        value = torch.tanh(self.fc_value2(v)).squeeze(-1)  # (B,)

        return policy_logits, value


# ===============================
# MCTS
# ===============================


class MCTS:
    def __init__(self, model: PolicyValueNet, sims: int = 200, cpuct: float = 1.5, device: torch.device = DEVICE):
        self.model = model
        self.sims = sims
        self.cpuct = cpuct
        self.device = device

        # state_key -> prior policy (Tensor 4096)
        self.P: Dict[str, torch.Tensor] = {}
        # state_key -> value from network
        self.V: Dict[str, float] = {}
        # (state_key, move_idx) -> visit count
        self.Nsa: Dict[Tuple[str, int], int] = {}
        # (state_key, move_idx) -> total value
        self.Wsa: Dict[Tuple[str, int], float] = {}
        # state_key -> total visits
        self.Ns: Dict[str, int] = {}

    @staticmethod
    def _state_key(board: chess.Board) -> str:
        return board.fen()

    @staticmethod
    def _terminal_value(board: chess.Board) -> float:
        result = board.result(claim_draw=True)
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        else:
            return 0.0

    def run(self, root_board: chess.Board) -> torch.Tensor:
        """
        Run MCTS simulations from root_board.
        Returns a 4096-dim policy vector (visit count distribution).
        """
        root_turn = root_board.turn

        for _ in range(self.sims):
            board = root_board.copy()
            path: List[Tuple[str, int, bool]] = []  # (state_key, move_idx, turn_at_state)

            # SELECTION & EXPANSION
            while True:
                s_key = self._state_key(board)

                if board.is_game_over():
                    v = self._terminal_value(board)
                    break

                if s_key not in self.P:
                    # expand leaf
                    x = board_to_tensor(board).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        logits, v_net = self.model(x)
                    logits = logits[0].cpu()
                    v = float(v_net[0].cpu())

                    # mask illegal moves
                    policy = torch.full((64 * 64,), float("-inf"))
                    legal_moves = list(board.legal_moves)
                    if not legal_moves:
                        self.P[s_key] = torch.zeros(64 * 64)
                        self.V[s_key] = 0.0
                        v = 0.0
                        break

                    for move in legal_moves:
                        idx = move_to_index(move)
                        policy[idx] = logits[idx]

                    policy = torch.softmax(policy, dim=0)
                    self.P[s_key] = policy
                    self.V[s_key] = v
                    break  # leaf node expanded

                # choose move via UCB
                policy = self.P[s_key]
                sum_N = self.Ns.get(s_key, 0) + 1
                best_score = -1e9
                best_move_idx: Optional[int] = None
                best_move: Optional[chess.Move] = None

                for move in board.legal_moves:
                    a_idx = move_to_index(move)
                    n_sa = self.Nsa.get((s_key, a_idx), 0)
                    w_sa = self.Wsa.get((s_key, a_idx), 0.0)
                    q = w_sa / n_sa if n_sa > 0 else 0.0
                    u = self.cpuct * policy[a_idx].item() * math.sqrt(sum_N) / (1 + n_sa)
                    score = q + u
                    if score > best_score:
                        best_score = score
                        best_move_idx = a_idx
                        best_move = move

                if best_move is None or best_move_idx is None:
                    v = 0.0
                    break

                path.append((s_key, best_move_idx, board.turn))
                board = board.copy()
                board.push(best_move)

            # BACKUP
            v_final = v
            for s_key, a_idx, turn_at_state in reversed(path):
                sign = 1.0 if turn_at_state == root_turn else -1.0
                self.Nsa[(s_key, a_idx)] = self.Nsa.get((s_key, a_idx), 0) + 1
                self.Wsa[(s_key, a_idx)] = self.Wsa.get((s_key, a_idx), 0.0) + sign * v_final
                self.Ns[s_key] = self.Ns.get(s_key, 0) + 1

        # build root policy from visit counts
        root_key = self._state_key(root_board)
        counts = torch.zeros(64 * 64)
        for move in root_board.legal_moves:
            idx = move_to_index(move)
            counts[idx] = self.Nsa.get((root_key, idx), 0)

        if counts.sum() > 0:
            pi = counts / counts.sum()
        else:
            pi = counts

        return pi


# ===============================
# Stockfish Distillation Dataset
# ===============================


class SFChessDataset(Dataset):
    def __init__(self, path: str):
        self.records: List[Dict] = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        board = chess.Board(rec["fen"])
        x = board_to_tensor(board)  # (17, 8, 8)

        policy_vec = torch.zeros(64 * 64, dtype=torch.float32)
        for uci, p in rec["policy"].items():
            move = chess.Move.from_uci(uci)
            policy_vec[move_to_index(move)] = float(p)

        value = torch.tensor(float(rec["value"]), dtype=torch.float32)
        return x, policy_vec, value


# ===============================
# Stockfish data generation
# ===============================


def random_board(max_plies: int = 20) -> chess.Board:
    board = chess.Board()
    n = random.randint(0, max_plies)
    for _ in range(n):
        if board.is_game_over():
            break
        move = random.choice(list(board.legal_moves))
        board.push(move)
    return board


def sf_policy_value(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    depth: int = 12,
    k: int = 4,
) -> Tuple[Dict[str, float], float]:
    """
    Ask Stockfish for top-k moves and an evaluation.
    Returns:
      policy: dict uci -> prob
      value: float in [-1, 1]
    """
    info = engine.analyse(board, limit=chess.engine.Limit(depth=depth), multipv=k)

    moves: List[chess.Move] = []
    scores: List[float] = []

    for entry in info:
        if "pv" not in entry or len(entry["pv"]) == 0:
            continue
        m = entry["pv"][0]
        s = entry["score"].white().score(mate_score=10000)
        if s is None:
            continue
        moves.append(m)
        scores.append(float(s))

    if not moves:
        # no moves or no scores; treat as draw-ish
        return {}, 0.0

    abs_scores = [abs(s) for s in scores]
    total = sum(abs_scores)
    if total <= 0.0:
        probs = [1.0 / len(moves)] * len(moves)
    else:
        probs = [a / total for a in abs_scores]

    policy = {m.uci(): p for m, p in zip(moves, probs)}

    raw_val = info[0]["score"].white().score(mate_score=10000)
    if raw_val is None:
        val = 0.0
    else:
        val = float(raw_val)
        # normalize to [-1, 1] with clipping
        val = max(-800.0, min(800.0, val)) / 800.0

    return policy, val


def generate_sf_data(
    engine_path: str,
    out_path: str,
    n_positions: int = 50000,
    depth: int = 10,
    multipv: int = 4,
    max_plies: int = 20,
) -> None:
    """
    Generate a JSONL file with Stockfish-labelled positions.
    """
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    with open(out_path, "w") as f:
        for i in range(n_positions):
            board = random_board(max_plies=max_plies)
            policy, value = sf_policy_value(engine, board, depth=depth, k=multipv)
            rec = {
                "fen": board.fen(),
                "policy": policy,
                "value": value,
            }
            f.write(json.dumps(rec) + "\n")
            if (i + 1) % 100 == 0:
                print(f"[gen] {i+1}/{n_positions}")

    engine.quit()
    print(f"Saved Stockfish data to {out_path}")


# ===============================
# Training loop (distillation)
# ===============================


def train_distill(
    data_path: str,
    out_model_path: str,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> None:
    dataset = SFChessDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = torch.nn.DataParallel(PolicyValueNet()).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for step, (x, pi_tgt, v_tgt) in enumerate(loader):
            x = x.to(DEVICE)
            pi_tgt = pi_tgt.to(DEVICE)
            v_tgt = v_tgt.to(DEVICE)

            opt.zero_grad()
            logits, v_pred = model(x)

            log_p = F.log_softmax(logits, dim=1)
            # handle cases where target policy is all-zero
            mask_nonzero = (pi_tgt.sum(dim=1) > 0)
            if mask_nonzero.any():
                log_p_sel = log_p[mask_nonzero]
                pi_sel = pi_tgt[mask_nonzero]
                policy_loss = F.kl_div(log_p_sel, pi_sel, reduction="batchmean")
            else:
                policy_loss = torch.tensor(0.0, device=DEVICE)

            value_loss = F.mse_loss(v_pred, v_tgt)

            loss = policy_loss + 0.5 * value_loss
            loss.backward()
            opt.step()

            running_loss += loss.item()

            if (step + 1) % 100 == 0:
                avg = running_loss / 100
                print(f"[epoch {epoch+1}] step {step+1} loss {avg:.4f}")
                running_loss = 0.0

    torch.save(model.state_dict(), out_model_path)
    print(f"Saved model to {out_model_path}")


# ===============================
# Playing: PGN → best move via MCTS
# ===============================


def get_move_from_pgn(pgn: str, model_path: str, sims: int = 200) -> str:
    """
    Load a trained model, reconstruct board from PGN, run MCTS, return best move in UCI.
    """
    import io

    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        raise ValueError("Invalid PGN")

    board = game.board()
    for move in game.mainline_moves():
        board.push(move)

    model = PolicyValueNet().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    mcts = MCTS(model, sims=sims, cpuct=1.5, device=DEVICE)
    pi = mcts.run(board)
    best_idx = int(torch.argmax(pi).item())
    move = index_to_move(best_idx)
    if move not in board.legal_moves:
        # fallback: choose any legal move
        move = list(board.legal_moves)[0]

    return move.uci()


# Optional: ChessHacks-style API
def get_move(pgn: str) -> str:
    """
    Thin wrapper so you can drop this into ChessHacks:
    - expects a file `policy_value_sf.pt` in the same folder.
    """
    model_path = os.path.join(os.path.dirname(__file__), "policy_value_sf.pt")
    return get_move_from_pgn(pgn, model_path=model_path, sims=200)


# ===============================
# Main CLI
# ===============================


def main():
    parser = argparse.ArgumentParser(description="Chess MCTS + Policy/Value Net (Stockfish-distilled).")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # generate data
    gen_p = subparsers.add_parser("gen-data", help="Generate Stockfish-labelled training data.")
    gen_p.add_argument("--engine", type=str, required=True, help="Path to Stockfish binary.")
    gen_p.add_argument("--out", type=str, default="sf_data.jsonl", help="Output JSONL file.")
    gen_p.add_argument("--n", type=int, default=50000, help="Number of positions.")
    gen_p.add_argument("--depth", type=int, default=10, help="Search depth.")
    gen_p.add_argument("--multipv", type=int, default=4, help="Number of principal variations.")
    gen_p.add_argument("--max-plies", type=int, default=20, help="Max random plies from startpos.")

    # train
    train_p = subparsers.add_parser("train", help="Train network on Stockfish data.")
    train_p.add_argument("--data", type=str, default="sf_data.jsonl", help="Path to JSONL training data.")
    train_p.add_argument("--out", type=str, default="policy_value_sf.pt", help="Output model path.")
    train_p.add_argument("--epochs", type=int, default=5)
    train_p.add_argument("--batch", type=int, default=64)
    train_p.add_argument("--lr", type=float, default=1e-3)

    # move from PGN
    pgn_p = subparsers.add_parser("pgn-move", help="Given a PGN string, output best move in UCI.")
    pgn_p.add_argument("--model", type=str, default="policy_value_sf.pt", help="Path to trained model.")
    pgn_p.add_argument("--pgn", type=str, required=True, help="PGN string.")
    pgn_p.add_argument("--sims", type=int, default=200, help="Number of MCTS simulations.")




    args = parser.parse_args()

    if args.cmd == "gen-data":
        generate_sf_data(
            engine_path=args.engine,
            out_path=args.out,
            n_positions=args.n,
            depth=args.depth,
            multipv=args.multipv,
            max_plies=args.max_plies,
        )
    elif args.cmd == "train":
        train_distill(
            data_path=args.data,
            out_model_path=args.out,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
        )
    elif args.cmd == "pgn-move":
        move_uci = get_move_from_pgn(args.pgn, args.model, sims=args.sims)
        print(move_uci)
        # eval all stockfish levels
    else:
        raise ValueError(f"Unknown command {args.cmd}")
    
    print("\n==== AUTO EVAL MODE: Evaluating Stockfish Levels 0–20 ====\n")

    eval_all_levels(
        model_path=args.model,
        sf_path=args.engine,
        games=args.games,
        sims=args.sims,
        device=DEVICE,
    )

if __name__ == "__main__":
    main()
