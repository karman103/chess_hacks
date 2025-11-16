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
        board_tensor = model2.board_to_tensor(ctx.board)
        
        board_tensor = board_tensor.unsqueeze(0)
        
        with torch.no_grad():
            policy_logits, value = model(board_tensor)
        
        policy_probs = F.softmax(policy_logits, dim=-1).squeeze(0)
        
        move_probs = {}
        for move in legal_moves:
            move_idx = model2.move_to_index(move)
            move_probs[move] = policy_probs[move_idx].item()
        
        total_prob = sum(move_probs.values())
        if total_prob > 0:
            move_probs = {move: prob / total_prob for move, prob in move_probs.items()}
        else:
            uniform_prob = 1.0 / len(legal_moves)
            move_probs = {move: uniform_prob for move in legal_moves}
        
        best_move = max(move_probs, key=move_probs.get)
   
        
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


