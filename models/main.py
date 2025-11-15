# src/main.py
from __future__ import annotations
import io
import os
import chess, chess.pgn
from .engine import TinyChessEngine

# Lazy singleton so hot-reloads don’t re-download/reload repeatedly
_ENGINE = None

def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        weights = os.path.join(os.path.dirname(__file__), "weights", "knight_edge.pt")
        # You can tune sims/cpuct here; higher sims = stronger but slower.
        _ENGINE = TinyChessEngine(
            weights_path=weights,
            width=64, blocks=8,
            sims=400, cpuct=1.5,
            device_str="cpu"  # use "mps" on Apple Silicon if you like
        )
    return _ENGINE

def _board_from_pgn(pgn_text: str) -> chess.Board:
    if not pgn_text.strip():
        return chess.Board()
    # Parse last game in the PGN (ChessHacks passes the current game)
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return chess.Board()
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
    return board

def get_move(pgn: str) -> str:
    """
    Called by the ChessHacks platform.
    Input: PGN string of the current game.
    Output: Move in UCI, e.g., 'e2e4'.
    """
    board = _board_from_pgn(pgn)
    engine = _get_engine()

    # Early exits (draw/resign policies can be added if desired)
    if board.is_game_over():
        # Fallback: just return any legal move string (platform expects a move)
        return next(iter(board.legal_moves)).uci()

    mv = engine.best_move(board)
    return mv.uci()
