#include "movegen.hpp"
#include <algorithm>

namespace chess {

std::vector<Move> MoveGen::generate_legal_moves(const Board& board) {
    std::vector<Move> moves;
    // TODO: Implement legal move generation for all pieces
    // Must include all chess rules: castling, en passant, promotion
    return moves;
}

std::vector<Move> MoveGen::generate_captures(const Board& board) {
    std::vector<Move> moves;
    // TODO: Implement capture-only move generation
    return moves;
}

bool MoveGen::is_legal(const Board& board, const Move& move) {
    // TODO: Validate if move is legal in current position
    return false;
}

bool MoveGen::is_in_check(const Board& board, Color side) {
    // TODO: Check if king is under attack
    return false;
}

bool MoveGen::is_checkmate(const Board& board) {
    // TODO: Check if position is checkmate
    return false;
}

bool MoveGen::is_stalemate(const Board& board) {
    // TODO: Check if position is stalemate
    return false;
}

bool MoveGen::is_insufficient_material(const Board& board) {
    // TODO: Check for insufficient material draw
    return false;
}

uint64_t MoveGen::perft(Board& board, int depth) {
    if (depth == 0) return 1;
    
    auto moves = generate_legal_moves(board);
    uint64_t nodes = 0;
    
    // TODO: Implement perft with make/unmake
    
    return nodes;
}

} // namespace chess
