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
model_path = os.path.join(os.path.dirname(__file__), "policy_model_fp16.pt")
model.eval()



@chess_manager.entrypoint
def test_func(ctx: GameContext):
    #

    print("Cooking move...")
    print(ctx.board.move_stack)
    time.sleep(0.1)

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")

    try:
        print("Step 3: Encoding board...")
        board_tensor = model2.board_to_tensor(ctx.board)
        print(f"Step 4: Board tensor shape: {board_tensor.shape}")
        
        board_tensor = board_tensor.unsqueeze(0)
        print(f"Step 5: After unsqueeze: {board_tensor.shape}")
        
        print("Step 6: Running model...")
        with torch.no_grad():
            policy_logits, value = model(board_tensor)
        print(f"Step 7: Got predictions. Policy shape: {policy_logits.shape}")
        
        print("Step 8: Applying softmax...")
        policy_probs = F.softmax(policy_logits, dim=-1).squeeze(0)
        print(f"Step 9: Policy probs shape: {policy_probs.shape}")
        
        print("Step 10: Getting move probabilities...")
        move_probs = {}
        for move in legal_moves:
            move_idx = model2.move_to_index(move)
            move_probs[move] = policy_probs[move_idx].item()
        print(f"Step 11: Extracted {len(move_probs)} move probabilities")
        
        print("Step 12: Normalizing...")
        total_prob = sum(move_probs.values())
        if total_prob > 0:
            move_probs = {move: prob / total_prob for move, prob in move_probs.items()}
        else:
            uniform_prob = 1.0 / len(legal_moves)
            move_probs = {move: uniform_prob for move in legal_moves}
        
        print("Step 13: Selecting best move...")
        best_move = max(move_probs, key=move_probs.get)
   
        
        print("Step 14: Logging probabilities...")
        ctx.logProbabilities(move_probs)
        
        print(f"Step 15: RETURNING MOVE: {best_move.uci()}")
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


