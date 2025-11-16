from .utils import chess_manager, GameContext
from chess import Move
from models import model2
import random
import time
import os
import torch
import torch.nn.functional as F



# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis
model = model2.PolicyValueNet()
model_path = os.path.join(os.path.dirname(__file__), "../policy_value_rl.pt")
model.load_state_dict(torch.load(model_path, map_location="cpu"))

model.to(model2.DEVICE)     # <-- ADD THIS
model.eval()




@chess_manager.entrypoint
def test_func(ctx: GameContext):

    print("Cooking move...")
    print(ctx.board.move_stack)
    time.sleep(0.1)

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")

    try:
        print("Step 3: Running MCTS...")

        # --- MCTS INFERENCE ---
        mcts = model2.MCTS(
            model=model,
            sims=800,
            device=model2.DEVICE
        )

        pi = mcts.run(ctx.board)  # tensor of visit counts (shape=[4672])
        print("Step 4: MCTS done.")

        # --- Pick best move by visits ---
        best_idx = torch.argmax(pi).item()
        best_move = model2.index_to_move(best_idx)

        # Safety: ensure best move is legal
        if best_move not in legal_moves:
            print("MCTS suggested illegal move — choosing fallback legal move.")
            best_move = legal_moves[0]

        # --- Convert MCTS distribution into move_probs dict ---
        move_probs = {}
        total_visits = pi.sum().item()
        for move in legal_moves:
            idx = model2.move_to_index(move)
            move_probs[move] = (pi[idx].item() / total_visits) if total_visits > 0 else 1.0 / len(legal_moves)

        print(f"Step 5: RETURNING MOVE FROM MCTS: {best_move.uci()}")

        # Required for leaderboard 🎯
        ctx.logProbabilities(move_probs)

        return best_move

    except Exception as e:
        print(f"ERROR at some step: {e}")
        import traceback
        traceback.print_exc()
        raise

@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass


