# main.py

import time
import math
import chess

from .evaluate import evaluate
from .pgn_to_fen import pgn_to_fen

INF = 1e9


def order_moves(board: chess.Board):
    """Simple move ordering: captures, promotions, then others."""
    moves = list(board.legal_moves)

    def score(move: chess.Move) -> int:
        s = 0
        if board.is_capture(move):
            s += 10
        if move.promotion:
            s += 8
        if board.gives_check(move):
            s += 5
        # prefer central moves a bit
        file_idx = chess.square_file(move.to_square)
        rank_idx = chess.square_rank(move.to_square)
        if 2 <= file_idx <= 5 and 2 <= rank_idx <= 5:
            s += 1
        return s

    moves.sort(key=score, reverse=True)
    return moves


class MinimaxSearcher:
    def __init__(self, max_depth: int = 3, time_limit: float | None = None):
        """
        Args:
            max_depth: maximum search depth in plies.
            time_limit: optional time limit in seconds for a single search.
                        If None, search will always go to full depth.
        """
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.start_time: float = 0.0
        self.nodes: int = 0
        self.time_up: bool = False
        # Simple transposition table: (key, depth) -> (depth, score)
        self.tt: dict[tuple, tuple[int, float]] = {}

    def _time_exceeded(self) -> bool:
        if self.time_limit is None:
            return False
        return time.time() - self.start_time >= self.time_limit

    def _static_eval(self, board: chess.Board) -> float:
        # evaluate() is from White's perspective in [-1, 1].
        # From side-to-move perspective, flip sign when it's Black to move.
        val = evaluate(board)
        return val if board.turn == chess.WHITE else -val

    def _hash_board(self, board: chess.Board) -> tuple:
        # A simple, reasonably unique key for a position.
        return (
            board.board_fen(),
            board.turn,
            board.castling_rights,
            board.ep_square,
        )

    def _negamax(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        if self._time_exceeded():
            self.time_up = True
            return self._static_eval(board)

        if depth == 0 or board.is_game_over():
            return self._static_eval(board)

        key = (self._hash_board(board), depth)
        if key in self.tt:
            return self.tt[key][1]

        max_score = -INF

        for move in order_moves(board):
            self.nodes += 1
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()

            if self.time_up:
                # Stop searching deeper; return best so far.
                return max_score if max_score != -INF else score

            if score > max_score:
                max_score = score

            if max_score > alpha:
                alpha = max_score
            if alpha >= beta:
                break

        self.tt[key] = (depth, max_score)
        return max_score

    def search(self, board: chess.Board):
        """
        Iterative deepening search.
        Returns (best_move, score_from_side_to_move_perspective).
        """
        self.start_time = time.time()
        self.nodes = 0
        self.time_up = False
        self.tt.clear()

        best_move: chess.Move | None = None
        best_score: float = self._static_eval(board)

        # If only one legal move, just play it.
        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 1:
            return legal_moves[0], best_score

        for depth in range(1, self.max_depth + 1):
            if self._time_exceeded():
                break

            current_best_move = None
            current_best_score = -INF

            for move in order_moves(board):
                if self._time_exceeded():
                    self.time_up = True
                    break

                self.nodes += 1
                board.push(move)
                score = -self._negamax(board, depth - 1, -INF, INF)
                board.pop()

                if self.time_up:
                    break

                if score > current_best_score:
                    current_best_score = score
                    current_best_move = move

            if self.time_up:
                break

            if current_best_move is not None:
                best_move = current_best_move
                best_score = current_best_score

        return best_move, best_score


def play_one_move_from_fen(fen: str, max_depth: int = 3, time_limit: float | None = None):
    board = chess.Board(fen)
    searcher = MinimaxSearcher(max_depth=max_depth, time_limit=time_limit)

    print("Position FEN:", fen)
    print("Side to move:", "White" if board.turn == chess.WHITE else "Black")
    print("Legal moves:")
    for move in board.legal_moves:
        print("-", move.uci())
    print()

    best_move, score = searcher.search(board)

    print("Best move:", best_move.uci() if best_move else None)
    print("Score (side-to-move pov):", score)
    print("Nodes searched:", searcher.nodes)
    return best_move, score


def main():
    # Example 1: start position
    board = chess.Board()
    print("Initial position:")
    print(board)
    print()

    # For bullet: depth 2–3, small time_limit (e.g., 0.1–0.3s)
    # For rapid: depth 4, larger time_limit (e.g., 1–2s)
    searcher = MinimaxSearcher(max_depth=3, time_limit=0.3)
    best_move, score = searcher.search(board)

    print("Best move:", best_move.uci() if best_move else None)
    print("Score:", score)
    print("Nodes searched:", searcher.nodes)

    # Example 2: from a PGN snippet
    example_pgn = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6"
    fen = pgn_to_fen(example_pgn)
    print("\nFrom PGN, final FEN:", fen)
    play_one_move_from_fen(fen, max_depth=3, time_limit=0.3)


if __name__ == "__main__":
    main()
