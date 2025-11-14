import chess
import numpy as np


def move_to_index(move):
    """
    Convert chess.Move to index (0 to 4671).
    
    Encoding scheme:
    - Regular moves: from_square * 64 + to_square = 0..4095
    - Promotions: 4096 + from_square * 16 + to_square * 4 + promo_type
      where promo_type: 0=knight, 1=bishop, 2=rook, 3=queen
    
    Args:
        move: chess.Move object
        
    Returns:
        Index (0-4671)
    """
    from_sq = move.from_square
    to_sq = move.to_square
    
    if move.promotion is None:
        # Regular move (including castling, en passant)
        return from_sq * 64 + to_sq
    else:
        # Promotion move
        # Map promotion piece: N=1->0, B=2->1, R=3->2, Q=4->3
        promo_idx = move.promotion - 1  # N,B,R,Q -> 0,1,2,3
        return 4096 + from_sq * 16 + (to_sq % 8) * 4 + promo_idx


def index_to_move(index, board=None):
    """
    Convert index back to chess.Move.
    
    Args:
        index: Move index (0-4671)
        board: Optional chess.Board for validation
        
    Returns:
        chess.Move object
    """
    if index < 4096:
        # Regular move
        from_sq = index // 64
        to_sq = index % 64
        move = chess.Move(from_sq, to_sq)
    else:
        # Promotion
        promo_index = index - 4096
        from_sq = promo_index // 16
        remainder = promo_index % 16
        to_file = remainder // 4
        promo_type = remainder % 4
        
        # Reconstruct to_square (we only encoded file, need to infer rank)
        # For white promotions: rank 7, for black: rank 0
        if board is not None:
            # Use board to determine color
            piece = board.piece_at(from_sq)
            if piece and piece.piece_type == chess.PAWN:
                if piece.color == chess.WHITE:
                    to_sq = 56 + to_file  # rank 7
                else:
                    to_sq = to_file  # rank 0
            else:
                raise ValueError(f"Invalid promotion from square {from_sq}")
        else:
            # Assume white promotion if no board provided
            to_sq = 56 + to_file
        
        # Map promo_type back to piece: 0->N, 1->B, 2->R, 3->Q
        promo_piece = promo_type + 1  # 0,1,2,3 -> N,B,R,Q (1,2,3,4)
        move = chess.Move(from_sq, to_sq, promotion=promo_piece)
    
    return move


def create_policy_mask(board):
    """
    Create a mask for legal moves in current position.
    
    Args:
        board: chess.Board object
        
    Returns:
        mask: (4672,) numpy array, 1.0 for legal moves, 0.0 otherwise
    """
    mask = np.zeros(4672, dtype=np.float32)
    
    for move in board.legal_moves:
        idx = move_to_index(move)
        mask[idx] = 1.0
    
    return mask


def moves_to_policy_vector(board, moves_with_probs):
    """
    Convert list of (move, probability) to full policy vector.
    
    Args:
        board: chess.Board object
        moves_with_probs: List of (chess.Move, float) tuples
        
    Returns:
        policy: (4672,) numpy array with probabilities
    """
    policy = np.zeros(4672, dtype=np.float32)
    
    for move, prob in moves_with_probs:
        idx = move_to_index(move)
        policy[idx] = prob
    
    return policy


def policy_vector_to_moves(policy, board, top_k=None):
    """
    Convert policy vector to list of (move, probability).
    
    Args:
        policy: (4672,) numpy array
        board: chess.Board object
        top_k: Return only top-k moves (None = all legal moves)
        
    Returns:
        List of (chess.Move, float) sorted by probability
    """
    legal_mask = create_policy_mask(board)
    masked_policy = policy * legal_mask
    
    # Get legal moves with their probabilities
    moves_probs = []
    for move in board.legal_moves:
        idx = move_to_index(move)
        prob = masked_policy[idx]
        moves_probs.append((move, prob))
    
    # Sort by probability
    moves_probs.sort(key=lambda x: x[1], reverse=True)
    
    if top_k is not None:
        moves_probs = moves_probs[:top_k]
    
    return moves_probs
