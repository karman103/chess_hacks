from .utils import chess_manager, GameContext
from chess import Move
from models import model2
import random
import time
import os
import torch
import torch.nn.functional as F


# Import your minimax components
from minmax import model3

# Initialize the searcher once
# Adjust these parameters based on your needs:
# - For bullet: max_depth=2-3, time_limit=0.1-0.3
# - For rapid: max_depth=4, time_limit=1-2
searcher = model3.MinimaxSearcher(max_depth=3, time_limit=0.5)

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    print("Cooking move...")
    print(ctx.board.move_stack)
    
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")
    
    # If only one legal move, just return it
    if len(legal_moves) == 1:
        ctx.logProbabilities({legal_moves[0]: 1.0})
        return legal_moves[0]
    
    try:
        print("Step 1: Running minimax search...")
        best_move, score = model3.searcher.search(ctx.board)
        print(f"Step 2: Best move found: {best_move.uci() if best_move else None}")
        print(f"Step 3: Score: {score}, Nodes searched: {searcher.nodes}")
        
        if best_move is None:
            # Fallback: pick a random legal move
            print("Warning: No move found, selecting random legal move")
            best_move = legal_moves[0]
        
        # Create probability distribution
        # Give the best move high probability, others low
        move_probs = {}
        for move in legal_moves:
            if move == best_move:
                move_probs[move] = 0.9  # High confidence in best move
            else:
                move_probs[move] = 0.1 / (len(legal_moves) - 1)
        
        # Normalize to ensure sum = 1.0
        total_prob = sum(move_probs.values())
        if total_prob > 0:
            move_probs = {move: prob / total_prob for move, prob in move_probs.items()}
        
        print("Step 4: Logging probabilities...")
        ctx.logProbabilities(move_probs)
        
        print(f"Step 5: RETURNING MOVE: {best_move.uci()}")
        return best_move
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to random move
        fallback_move = legal_moves[0]
        ctx.logProbabilities({fallback_move: 1.0})
        return fallback_move

@chess_manager.reset
def reset_func(ctx: GameContext):
    # Clear the transposition table for a fresh game
    global searcher
    searcher.tt.clear()
    searcher.nodes = 0
    searcher.time_up = False
    print("Game reset: cleared transposition table")