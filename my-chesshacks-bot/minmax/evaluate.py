# evaluate.py
"""
NNUE-style evaluator for chess positions.

- Encodes a chess.Board into a 8x8x14 tensor with the following planes:
    0-5 : piece-type planes (P, N, B, R, Q, K), colour-agnostic
    6-7 : colour planes (white piece, black piece)
    8   : side-to-move plane (1 for white to move, 0 for black)
    9   : white kingside castling right (all ones if available, else zeros)
    10  : white queenside castling right
    11  : black kingside castling right
    12  : black queenside castling right
    13  : en-passant plane (1 at the ep target square if any)

Flattened to a 896-dim vector (8*8*14) and fed into a small fully-connected NN.

The model weights are loaded from nnue_cpu.pth and cached globally.
"""

import torch
import torch.nn as nn
import numpy as np
import chess

MODEL_PATH = "nnue_cpu.pth"
DEVICE = "cpu"

# Limit PyTorch internal threading – helps latency on small models.
torch.set_num_threads(1)


###########################################################
# 1. NNUE Model  (must match the training architecture)
###########################################################
class NNUE(nn.Module):
    def __init__(self, input_dim: int = 8 * 8 * 14):
        super().__init__()
        # Adjust sizes if your training script used a different architecture.
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 896]
        x = x.float()
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)


_model: NNUE | None = None


def load_model() -> NNUE:
    """
    Lazy-load the NNUE model and cache it.
    Supports either a raw state_dict or {'model_state_dict': ...}.
    """
    global _model
    if _model is not None:
        return _model

    model = NNUE()
    state = torch.load(MODEL_PATH, map_mode=DEVICE) if hasattr(torch, "load") else torch.load(MODEL_PATH, map_location=DEVICE)

    # (some environments use map_mode param name, some use map_location)
    # If the above line errors in your local env, change it to:
    # state = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()
    _model = model
    return model


###########################################################
# 2. FEN -> 8x8x14 tensor
###########################################################
PIECE_TYPES = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def fen_to_tensor(fen: str) -> torch.Tensor:
    """
    Convert a FEN string to a flattened 8x8x14 tensor.

    Coordinate convention:
        - planes[rank_index, file_index, :] where
          rank_index = 0 is rank 8 (top from White's perspective),
          rank_index = 7 is rank 1 (bottom),
          file_index = 0 is file 'a', file_index = 7 is file 'h'.
    """
    board = chess.Board(fen)

    planes = np.zeros((8, 8, 14), dtype=np.float32)

    # 0–5: piece-type, 6–7: colour
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        file_idx = chess.square_file(square)        # 0..7 => a..h
        rank_idx = chess.square_rank(square)        # 0..7 => 1..8
        row = 7 - rank_idx                          # 0 => rank 8, 7 => rank 1
        col = file_idx

        type_plane = PIECE_TYPES[piece.piece_type]
        planes[row, col, type_plane] = 1.0

        if piece.color == chess.WHITE:
            planes[row, col, 6] = 1.0
        else:
            planes[row, col, 7] = 1.0

    # 8: side to move (all ones if white to move, else zeros)
    if board.turn == chess.WHITE:
        planes[:, :, 8] = 1.0

    # 9–12: castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[:, :, 9] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[:, :, 10] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[:, :, 11] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[:, :, 12] = 1.0

    # 13: en-passant square (one-hot)
    ep_square = board.ep_square
    if ep_square is not None:
        file_idx = chess.square_file(ep_square)
        rank_idx = chess.square_rank(ep_square)
        row = 7 - rank_idx
        col = file_idx
        planes[row, col, 13] = 1.0

    # Flatten to [896]
    flat = planes.reshape(-1)
    return torch.from_numpy(flat)


###########################################################
# 3. Public API — evaluate a chess.Board
###########################################################
@torch.no_grad()
def evaluate(board: chess.Board) -> float:
    """
    Evaluate board position from White's perspective.

    Returns:
        float in approximately [-1, +1],
        where positive is good for White and negative for Black.
        (This assumes you trained the network on normalized Stockfish evals.)
    """
    model = load_model()
    x = fen_to_tensor(board.fen()).to(DEVICE).unsqueeze(0)  # [1, 896]
    pred = model(x).item()
    return float(pred)


if __name__ == "__main__":
    # Quick manual test on the starting position.
    start_board = chess.Board()
    print("Start position eval (White pov):", evaluate(start_board))
