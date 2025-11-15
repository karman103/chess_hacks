import chess
import chess.engine
import torch
import os
os.environ["PYTORCH_JIT_USE_NNPACK"] = "0"

import random
import math
import argparse

from model2 import PolicyValueNet, MCTS, board_to_tensor, index_to_move


###############################################
# Move selection using your MCTS model
###############################################
def model_move(board, model, sims=64, device="cpu"):
    mcts = MCTS(model, sims=sims, cpuct=1.5, device=device)
    pi = mcts.run(board)
    move_idx = int(torch.argmax(pi))
    move = index_to_move(move_idx)

    if move not in board.legal_moves:
        move = random.choice(list(board.legal_moves))
    return move


###############################################
# Play ONE game vs Stockfish
###############################################
def play_game(model, sf_path, model_white=True, sims=64, device="cpu"):
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(sf_path)

    while not board.is_game_over():
        if (board.turn == chess.WHITE and model_white) or \
           (board.turn == chess.BLACK and not model_white):
            # model plays
            move = model_move(board, model, sims=sims, device=device)
        else:
            # stockfish plays
            result = engine.play(board, chess.engine.Limit(time=0.05))
            move = result.move

        board.push(move)

    engine.quit()
    return board.result()  # "1-0", "0-1", "1/2-1/2"


###############################################
# Compute Elo difference
###############################################
def elo_from_results(results):
    wins = results["1-0"]
    losses = results["0-1"]
    draws = results["1/2-1/2"]
    N = wins + losses + draws

    if N == 0:
        return 0

    score = (wins + 0.5 * draws) / N

    # Avoid log(0)
    score = max(1e-6, min(1 - 1e-6, score))

    elo = -400 * math.log10((1 / score) - 1)
    return elo


###############################################
# Main
###############################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="policy_value_sf.pt")
    parser.add_argument("--stockfish", type=str, default="/usr/games/stockfish")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--sims", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = PolicyValueNet().to(device)

    raw_state = torch.load(args.model, map_location=device)

    # Fix for DataParallel checkpoints
    fixed_state = {k.replace("module.", ""): v for k, v in raw_state.items()}

    model.load_state_dict(fixed_state)

    model.eval()

    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}

    for i in range(args.games):
        model_white = (i % 2 == 0)

        res = play_game(
            model=model,
            sf_path=args.stockfish,
            model_white=model_white,
            sims=args.sims,
            device=device,
        )
        results[res] += 1

        print(f"Game {i+1}/{args.games}: {res}")

    print("\n===== RESULTS =====")
    print(results)

    elo = elo_from_results(results)
    print(f"\nEstimated Elo difference vs Stockfish Level 1: {elo:.1f} Elo")

    # Approximate SF Level 1 Elo anchor
    stockfish_level1_elo = 1150
    model_elo = stockfish_level1_elo + elo
    print(f"Approximate Model Elo: {model_elo:.1f}")


if __name__ == "__main__":
    main()
