# pgn_to_fen.py

import io
import chess
import chess.pgn


def _apply_movelist(board: chess.Board, moves_text: str) -> chess.Board:
    """
    Fallback parser for plain movelists without PGN headers.

    Example accepted formats:
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6"
        "e4 e5 Nf3 Nc6 Bb5 a6"
    """
    tokens = moves_text.replace("\n", " ").split()

    for tok in tokens:
        # Skip move numbers like "1.", "1...", "23."
        if tok.endswith(".") or tok.endswith("..."):
            continue
        # Skip result markers
        if tok in ("1-0", "0-1", "1/2-1/2", "*"):
            break

        try:
            board.push_san(tok)
        except ValueError:
            # Ignore anything that cannot be parsed as SAN (comments, NAGs, etc.)
            continue

    return board


def pgn_to_fen(pgn_text: str) -> str:
    """
    Convert a PGN game or movelist into the final FEN.

    Supports:
      - Full PGNs with headers
      - PGNs without headers
      - Plain movelists
      - Partial games
    """
    pgn_text = pgn_text.strip()
    if not pgn_text:
        return chess.Board().fen()

    # First try using the PGN parser
    pgn_io = io.StringIO(pgn_text)
    game = chess.pgn.read_game(pgn_io)

    if game is not None:
        board = game.end().board()
        return board.fen()

    # Fallback to a simple movelist interpretation
    board = chess.Board()
    board = _apply_movelist(board, pgn_text)
    return board.fen()


if __name__ == "__main__":
    example_pgn = """
    1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. O-O Nf6
    """
    final_fen = pgn_to_fen(example_pgn)
    print("Final FEN:", final_fen)
