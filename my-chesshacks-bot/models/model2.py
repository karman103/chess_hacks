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

from multiprocessing import Process, Value, Lock
from math import ceil


def _sf_worker(
    worker_id: int,
    n_positions: int,
    engine_path: str,
    depth: int,
    multipv: int,
    max_plies: int,
    out_path: str,
    global_counter: "Value",
    counter_lock: "Lock",
    total_positions: int,
):
    """
    Worker for parallel Stockfish data gen.
    Each worker:
      - runs its own Stockfish engine
      - writes to its own temp file
      - updates a shared global counter for [gen] progress prints
    """
    print(f"[WORKER {worker_id}] Starting, n_positions={n_positions}", flush=True)

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

            # 🔥 global progress
            with counter_lock:
                global_counter.value += 1
                # print every 100 positions globally (tweak as you like)
                if global_counter.value % 100 == 0:
                    print(f"[gen] {global_counter.value}/{total_positions}", flush=True)

    engine.quit()
    print(f"[WORKER {worker_id}] Done.", flush=True)


# ===============================
# Device selection
# ===============================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"[INFO] Using device: {DEVICE}")


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
# Batched inference manager
# ===============================


class BatchManager:
    """
    Simple local batching helper for MCTS:
      - collect (state_key, tensor) pairs
      - run them through the model in one big batch
      - store outputs in a dict keyed by state_key
    """

    def __init__(self, model: PolicyValueNet, device: torch.device, batch_size: int = 32):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.pending: List[Tuple[str, torch.Tensor]] = []
        self.outputs: Dict[str, Tuple[torch.Tensor, float]] = {}

    def request(self, state_key: str, tensor: torch.Tensor) -> None:
        """
        Queue a single (state_key, tensor) for batched evaluation.
        tensor is (17, 8, 8) on CPU.
        """
        self.pending.append((state_key, tensor))
        if len(self.pending) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """
        Run all pending tensors through the network in one batch.
        """
        if not self.pending:
            return

        keys, tensors = zip(*self.pending)
        batch = torch.stack(tensors).to(self.device)  # (B,17,8,8)

        with torch.no_grad():
            logits_batch, values_batch = self.model(batch)

        logits_batch = logits_batch.cpu()
        values_batch = values_batch.cpu()

        for i, key in enumerate(keys):
            self.outputs[key] = (logits_batch[i], float(values_batch[i].item()))

        self.pending = []

    def get(self, state_key: str) -> Optional[Tuple[torch.Tensor, float]]:
        """
        Retrieve and remove stored (logits, value) for a given state_key.
        """
        return self.outputs.pop(state_key, None)


# ===============================
# MCTS
# ===============================


class MCTS:
    def __init__(
        self,
        model: PolicyValueNet,
        sims: int = 200,
        cpuct: float = 1.5,
        device: torch.device = DEVICE,
        batch_size: int = 32,
    ):
        self.model = model
        self.sims = sims
        self.cpuct = cpuct
        self.device = device
        self.batch_size = batch_size

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

    def _backup(self, path: List[Tuple[str, int, bool]], v_final: float, root_turn: bool) -> None:
        """
        Backup value v_final along the path of (state_key, action_idx, turn_at_state).
        """
        for s_key, a_idx, turn_at_state in reversed(path):
            sign = 1.0 if turn_at_state == root_turn else -1.0
            self.Nsa[(s_key, a_idx)] = self.Nsa.get((s_key, a_idx), 0) + 1
            self.Wsa[(s_key, a_idx)] = self.Wsa.get((s_key, a_idx), 0.0) + sign * v_final
            self.Ns[s_key] = self.Ns.get(s_key, 0) + 1

    def run(self, root_board: chess.Board) -> torch.Tensor:
        """
        Run MCTS simulations from root_board.
        Uses batched NN inference for leaf evaluations.
        Returns a 4096-dim policy vector (visit count distribution).
        """
        root_turn = root_board.turn
        batch_mgr = BatchManager(self.model, self.device, batch_size=self.batch_size)

        sims_done = 0
        while sims_done < self.sims:
            # Selection for up to batch_size simulations
            pending_leaves: List[Tuple[str, chess.Board, List[Tuple[str, int, bool]]]] = []

            batch_quota = min(self.batch_size, self.sims - sims_done)
            for _ in range(batch_quota):
                board = root_board.copy()
                path: List[Tuple[str, int, bool]] = []

                while True:
                    s_key = self._state_key(board)

                    # Terminal node
                    if board.is_game_over():
                        v = self._terminal_value(board)
                        self._backup(path, v, root_turn)
                        break

                    # Leaf node: queue for batched NN evaluation
                    if s_key not in self.P:
                        tensor = board_to_tensor(board)  # (17,8,8) on CPU
                        batch_mgr.request(s_key, tensor)
                        pending_leaves.append((s_key, board.copy(), path))
                        break

                    # Selection: choose move via UCB
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
                        # Should not generally happen, but treat as draw
                        v = 0.0
                        self._backup(path, v, root_turn)
                        break

                    path.append((s_key, best_move_idx, board.turn))
                    board.push(best_move)

                sims_done += 1

            # Run batched inference on all leaf nodes gathered in this mini-batch
            batch_mgr.flush()

            # Expansion + backup for all pending leaves
            for s_key, leaf_board, path in pending_leaves:
                out = batch_mgr.get(s_key)
                if out is None:
                    # Shouldn't happen; skip
                    continue
                logits, v = out

                # Mask illegal moves
                policy = torch.full((64 * 64,), float("-inf"))
                legal_moves = list(leaf_board.legal_moves)
                if not legal_moves:
                    # No legal moves from this state, treat as draw-ish leaf
                    self.P[s_key] = torch.zeros(64 * 64)
                    self.V[s_key] = 0.0
                    self._backup(path, 0.0, root_turn)
                    continue

                for move in legal_moves:
                    idx = move_to_index(move)
                    policy[idx] = logits[idx]

                policy = torch.softmax(policy, dim=0)
                self.P[s_key] = policy
                self.V[s_key] = v

                self._backup(path, v, root_turn)

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
# Generic JSONL Dataset
# (works for both Stockfish and self-play)
# ===============================


class SFChessDataset(Dataset):
    """
    JSONL format:
    {
        "fen": <str>,
        "policy": { "<uci>": prob, ... },
        "value": <float>   # Stockfish eval in [-1,1] OR game result z in [-1,1]
    }
    """
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
# Stockfish data generation (distillation)
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

    import numpy as np

    # Convert Stockfish centipawn evals into logits
    scores_np = np.array(scores, dtype=np.float32)

    # Temperature — lower = sharper, higher = smoother
    T = 200.0  # good default for Stockfish multipv

    # Softmax over (-score/T) so best score → highest prob
    logits = torch.tensor(scores_np) * (-1.0 / T)
    probs = torch.softmax(logits, dim=0).tolist()

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
    print(f"[GEN] Using engine: {engine_path}")
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
            if (i + 1) % 50 == 0:
                print(f"[gen] {i+1}/{n_positions}")

    engine.quit()
    print(f"[GEN] Saved Stockfish data to {out_path}")


def generate_sf_data_parallel(
    engine_path: str,
    out_path: str,
    n_positions: int = 50000,
    depth: int = 10,
    multipv: int = 4,
    max_plies: int = 20,
    num_workers: Optional[int] = None,
) -> None:
    """
    Parallel Stockfish data generation using separate processes.
    Each worker writes to its own temp file; main process merges them.
    """
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 1) - 1)

    print(f"[GEN-PAR] Using engine: {engine_path}")
    print(f"[GEN-PAR] n_positions={n_positions}, workers={num_workers}")

    # Split positions across workers as evenly as possible
    base = n_positions // num_workers
    rem = n_positions % num_workers
    counts = [base + (1 if i < rem else 0) for i in range(num_workers)]
    counts = [c for c in counts if c > 0]

    # Shared counter and lock for global progress
    global_counter = Value("i", 0)
    counter_lock = Lock()

    procs: List[Process] = []
    tmp_paths: List[str] = []

    for wid, n_pos in enumerate(counts):
        tmp_out = f"{out_path}.part{wid}"
        tmp_paths.append(tmp_out)

        p = Process(
            target=_sf_worker,
            args=(
                wid,
                n_pos,
                engine_path,
                depth,
                multipv,
                max_plies,
                tmp_out,
                global_counter,
                counter_lock,
                n_positions,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # Merge partial files into final output
    with open(out_path, "w") as fout:
        for tmp in tmp_paths:
            if not os.path.exists(tmp):
                continue
            with open(tmp, "r") as fin:
                for line in fin:
                    fout.write(line)
            os.remove(tmp)

    print(f"[GEN-PAR] Saved Stockfish data to {out_path}")


# ===============================
# AlphaZero-style self-play generation
# ===============================


def result_to_value(result: str, player_to_move_was_white: bool) -> float:
    """Map game result string to +1/-1/0 from the POV of a given player."""
    if result == "1-0":
        return 1.0 if player_to_move_was_white else -1.0
    elif result == "0-1":
        return -1.0 if player_to_move_was_white else 1.0
    else:
        return 0.0


def self_play_one_game(
    model: PolicyValueNet,
    sims: int = 200,
    max_moves: int = 512,
) -> List[Dict]:
    """
    Play one self-play game using MCTS and the current network.
    Returns a list of JSON-serializable records:
      {"fen": ..., "policy": {...}, "value": z}
    where z is the final game result from the POV of the player to move.
    """
    model.eval()
    board = chess.Board()
    history: List[Tuple[str, Dict[str, float], bool]] = []  # (fen, policy_dict, player_is_white)

    move_count = 0
    while not board.is_game_over() and move_count < max_moves:
        mcts = MCTS(model, sims=sims, cpuct=1.5, device=DEVICE)
        pi = mcts.run(board)  # (4096,)
                # === FIX 2: Dirichlet noise at root, adds exploration to avoid draw loops ===
        if move_count == 0:  # Only on root each turn
            legal = list(board.legal_moves)
            if len(legal) > 1:
                import numpy as np
                noise = np.random.dirichlet([0.3] * len(legal))
                # mix noise into MCTS distribution
                for i, move in enumerate(legal):
                    idx = move_to_index(move)
                    pi[idx] = 0.75 * pi[idx] + 0.25 * noise[i]

        # Build dict of legal moves -> prob (from MCTS visit distribution)
        policy_dict: Dict[str, float] = {}
        legal_moves = list(board.legal_moves)
        move_probs: List[Tuple[chess.Move, float]] = []
        for move in legal_moves:
            idx = move_to_index(move)
            p = float(pi[idx].item())
            if p > 0.0:
                policy_dict[move.uci()] = p
            move_probs.append((move, p))

        # Normalize over legal moves for sampling/argmax
        total_p = sum(max(p, 0.0) for _, p in move_probs)
        if total_p <= 0.0:
            # fallback: uniform
            probs = [1.0 / len(move_probs)] * len(move_probs)
        else:
            probs = [max(p, 0.0) / total_p for (_, p) in move_probs]

        # Choose move: argmax for now (you could sample for more exploration)
        best_idx = max(range(len(move_probs)), key=lambda i: probs[i])
        # chosen_move = move_probs[best_idx][0]
        import random
        chosen_move = random.choices(
            [m for (m,p) in move_probs],
            weights=[max(p, 1e-6) for (m,p) in move_probs],
            k=1
        )[0]

        # Record state before making the move
        history.append((board.fen(), policy_dict, board.turn))

        board.push(chosen_move)
        move_count += 1

    result = board.result(claim_draw=True)
    print(f"[SELFPLAY] Game finished after {move_count} moves with result {result}")

    records: List[Dict] = []
    for fen, policy_dict, player_is_white in history:
        z = result_to_value(result, player_is_white)
        records.append(
            {
                "fen": fen,
                "policy": policy_dict,
                "value": z,
            }
        )

    return records


def train_on_selfplay_batch(model, batch, opt):
    """
    batch = list of records like:
       {"fen": ..., "policy": {...}, "value": z}
    This performs 1 gradient update like AlphaZero.
    """
    model.train()

    xs = []
    pi_targets = []
    v_targets = []

    for rec in batch:
        board = chess.Board(rec["fen"])
        xs.append(board_to_tensor(board))

        # policy target -> vector of 4096
        pi_vec = torch.zeros(4096)
        for uci, p in rec["policy"].items():
            move = chess.Move.from_uci(uci)
            idx = move_to_index(move)
            pi_vec[idx] = p
        pi_targets.append(pi_vec)

        v_targets.append(torch.tensor(rec["value"], dtype=torch.float32))

    x = torch.stack(xs).to(DEVICE)
    pi_t = torch.stack(pi_targets).to(DEVICE)
    v_t = torch.stack(v_targets).to(DEVICE)

    opt.zero_grad()

    logits, v_pred = model(x)
    log_p = F.log_softmax(logits, dim=1)

    # KL for policy
    policy_loss = F.kl_div(log_p, pi_t, reduction="batchmean")

    # MSE for value
    value_loss = F.mse_loss(v_pred, v_t)

    loss = policy_loss + 0.5 * value_loss
    loss.backward()
    opt.step()

    return loss.item(), policy_loss.item(), value_loss.item()


def selfplay_learn(
    model_path: Optional[str],
    out_model_path: str,
    n_games: int = 100,
    sims: int = 200,
    max_moves: int = 512,
    lr: float = 1e-3,
):
    """
    Full AlphaZero-style reinforcement learning:
      - self-play
      - train on self-play every game
      - update model continuously
    """
    model = PolicyValueNet().to(DEVICE)

    if model_path is not None and os.path.exists(model_path):
        print(f"[RL] Loading starting model from {model_path}")
        state = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state)
    else:
        print("[RL] Starting from scratch (no model found).")

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for g in range(n_games):
        print(f"[RL] ===== Game {g+1}/{n_games} =====")
        game_records = self_play_one_game(model, sims=sims, max_moves=max_moves)

        loss, ploss, vloss = train_on_selfplay_batch(model, game_records, opt)

        print(f"[RL] Game {g+1} finished: positions={len(game_records)}")
        print(f"[RL] Loss total={loss:.4f}  policy={ploss:.4f}  value={vloss:.4f}")

        # save checkpoint each game
        torch.save(model.state_dict(), out_model_path)

    print(f"[RL] Finished. Final model saved to {out_model_path}")


# ===============================
# Training loop (supervised on JSONL)
# Works for both Stockfish and self-play data.
# ===============================


def train_supervised(
    data_path: str,
    out_model_path: str,
    input_model_path: Optional[str] = None,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> None:
    dataset = SFChessDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = PolicyValueNet().to(DEVICE)

    # 🔥 NEW: optional checkpoint loading
    if input_model_path is not None:
        if os.path.exists(input_model_path):
            print(f"[TRAIN] Loading checkpoint from {input_model_path}")
            state = torch.load(input_model_path, map_location=DEVICE)
            model.load_state_dict(state)
        else:
            print(f"[TRAIN] WARNING: input checkpoint {input_model_path} not found, training from scratch.")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"[TRAIN] Training on {len(dataset)} positions from {data_path}")
    print(f"[TRAIN] Saving to {out_model_path}")
    print(f"[TRAIN] Using Adam, lr={lr}")

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
    print(f"[TRAIN] Saved model to {out_model_path}")


# Backwards compatibility name
train_distill = train_supervised


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
    parser = argparse.ArgumentParser(
        description="Chess MCTS + Policy/Value Net.\n"
                    "Supports Stockfish distillation + AlphaZero-style self-play."
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # generate data (Stockfish)
    gen_p = subparsers.add_parser("gen-data", help="Generate Stockfish-labelled training data.")
    gen_p.add_argument("--engine", type=str, default='/usr/games/stockfish', help="Path to Stockfish binary.")
    gen_p.add_argument("--out", type=str, default="sf_data.jsonl", help="Output JSONL file.")
    gen_p.add_argument("--n", type=int, default=50000, help="Number of positions.")
    gen_p.add_argument("--depth", type=int, default=10, help="Search depth.")
    gen_p.add_argument("--multipv", type=int, default=4, help="Number of principal variations.")
    gen_p.add_argument("--max-plies", type=int, default=20, help="Max random plies from startpos.")
    # in main(), in gen-data parser:
    gen_p.add_argument("--workers", type=int, default=1, help="Number of parallel workers.")

    # train (supervised: works for Stockfish or self-play JSONL)
    train_p = subparsers.add_parser("train", help="Train network on JSONL data (Stockfish or self-play).")
    train_p.add_argument("--data", type=str, default="sf_data.jsonl", help="Path to JSONL training data.")
    train_p.add_argument("--out", type=str, default="policy_value_sf.pt", help="Output model path.")
    train_p.add_argument("--input", type=str, default=None, help="Optional checkpoint to continue training.")
    train_p.add_argument("--epochs", type=int, default=5)
    train_p.add_argument("--batch", type=int, default=64)
    train_p.add_argument("--lr", type=float, default=1e-3)

    # self-play (AlphaZero style)
    # AlphaZero selfplay DATA ONLY
    sp_p = subparsers.add_parser("selfplay-data", help="Generate pure self-play data without learning.")
    ...

    # AlphaZero full reinforcement learning
    rl_p = subparsers.add_parser("selfplay", help="Self-play + online learning (AlphaZero RL).")
    rl_p.add_argument("--model", type=str, default=None)
    rl_p.add_argument("--out", type=str, default="policy_value_rl.pt")
    rl_p.add_argument("--games", type=int, default=100)
    rl_p.add_argument("--sims", type=int, default=200)
    rl_p.add_argument("--max-moves", type=int, default=512)
    rl_p.add_argument("--lr", type=float, default=1e-3)

    # move from PGN
    pgn_p = subparsers.add_parser("pgn-move", help="Given a PGN string, output best move in UCI.")
    pgn_p.add_argument("--model", type=str, default="policy_value_sf.pt", help="Path to trained model.")
    pgn_p.add_argument("--pgn", type=str, required=True, help="PGN string.")
    pgn_p.add_argument("--sims", type=int, default=200, help="Number of MCTS simulations.")

    args = parser.parse_args()

    if args.cmd == "gen-data":
        if args.workers == 1:
            generate_sf_data(
                engine_path=args.engine,
                out_path=args.out,
                n_positions=args.n,
                depth=args.depth,
                multipv=args.multipv,
                max_plies=args.max_plies,
            )
        else:
            generate_sf_data_parallel(
                engine_path=args.engine,
                out_path=args.out,
                n_positions=args.n,
                depth=args.depth,
                multipv=args.multipv,
                max_plies=args.max_plies,
                num_workers=args.workers,
            )

    elif args.cmd == "train":
        train_supervised(
            data_path=args.data,
            out_model_path=args.out,
            input_model_path=args.input,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
        )

    elif args.cmd == "selfplay":
        selfplay_learn(
            model_path=args.model,
            out_model_path=args.out,
            n_games=args.games,
            sims=args.sims,
            max_moves=args.max_moves,
            lr=args.lr,
        )

    elif args.cmd == "pgn-move":
        move_uci = get_move_from_pgn(args.pgn, args.model, sims=args.sims)
        print(move_uci)
    else:
        raise ValueError(f"Unknown command {args.cmd}")


if __name__ == "__main__":
    main()
