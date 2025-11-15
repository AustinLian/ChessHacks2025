import numpy as np
import chess


def fen_to_planes(fen_string):
    """
    Convert FEN string to 27-plane board representation.
    
    Planes:
    - 0-5: White pieces (P, N, B, R, Q, K)
    - 6-11: Black pieces (P, N, B, R, Q, K)
    - 12-19: Board history (last 4 positions, 2 planes each)
    - 20: White castling kingside
    - 21: White castling queenside
    - 22: Black castling kingside
    - 23: Black castling queenside
    - 24: En passant square
    - 25: Side to move (1 = white, 0 = black)
    - 26: Move count / 50-move rule progress
    
    Args:
        fen_string: FEN position string
        
    Returns:
        planes: (27, 8, 8) numpy array
    """
    board = chess.Board(fen_string)
    planes = np.zeros((27, 8, 8), dtype=np.float32)
    
    # Piece planes (0-11)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        rank = square // 8
        file = square % 8
        
        piece_type = piece.piece_type - 1  # 0-5 for P,N,B,R,Q,K
        color_offset = 0 if piece.color == chess.WHITE else 6
        plane_idx = color_offset + piece_type
        
        planes[plane_idx, rank, file] = 1.0
    
    # History planes (12-19) - TODO: requires position history
    # For now, leave as zeros
    
    # Castling rights (20-23)
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[20, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[21, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[22, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[23, :, :] = 1.0
    
    # En passant (24)
    if board.ep_square is not None:
        rank = board.ep_square // 8
        file = board.ep_square % 8
        planes[24, rank, file] = 1.0
    
    # Side to move (25)
    if board.turn == chess.WHITE:
        planes[25, :, :] = 1.0
    
    # Move count / 50-move rule (26)
    planes[26, :, :] = board.halfmove_clock / 100.0  # Normalize
    
    return planes


def planes_to_fen(planes):
    """
    Convert 27-plane representation back to FEN (approximate).
    
    Note: History planes and exact move counts may not be recoverable.
    
    Args:
        planes: (27, 8, 8) numpy array
        
    Returns:
        fen_string: FEN position string
    """
    # TODO: Implement reverse conversion
    # This is mainly for debugging/visualization
    raise NotImplementedError("planes_to_fen not yet implemented")


def augment_planes(planes, flip_horizontal=False, rotate_colors=False):
    """
    Apply data augmentation to board planes.
    
    Args:
        planes: (27, 8, 8) numpy array
        flip_horizontal: Flip board left-right
        rotate_colors: Swap white/black pieces
        
    Returns:
        Augmented planes
    """
    result = planes.copy()
    
    if flip_horizontal:
        result = np.flip(result, axis=2).copy()
    
    if rotate_colors:
        # Swap white/black piece planes (0-5 <-> 6-11)
        temp = result[0:6].copy()
        result[0:6] = result[6:12]
        result[6:12] = temp
        
        # Swap castling rights (20-21 <-> 22-23)
        temp = result[20:22].copy()
        result[20:22] = result[22:24]
        result[22:24] = temp
        
        # Flip side to move (25)
        result[25] = 1.0 - result[25]
    
    return result
