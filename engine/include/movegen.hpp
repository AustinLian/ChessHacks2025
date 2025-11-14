#pragma once

#include "board.hpp"
#include "move.hpp"
#include <vector>

namespace chess {

// Move generation
class MoveGen {
public:
    static std::vector<Move> generate_legal_moves(const Board& board);
    static std::vector<Move> generate_captures(const Board& board);
    static bool is_legal(const Board& board, const Move& move);
    static bool is_in_check(const Board& board, Color side);
    static bool is_checkmate(const Board& board);
    static bool is_stalemate(const Board& board);
    static bool is_insufficient_material(const Board& board);
    
    // Perft for testing
    static uint64_t perft(Board& board, int depth);
    
private:
    static void generate_pawn_moves(const Board& board, std::vector<Move>& moves);
    static void generate_knight_moves(const Board& board, std::vector<Move>& moves);
    static void generate_bishop_moves(const Board& board, std::vector<Move>& moves);
    static void generate_rook_moves(const Board& board, std::vector<Move>& moves);
    static void generate_queen_moves(const Board& board, std::vector<Move>& moves);
    static void generate_king_moves(const Board& board, std::vector<Move>& moves);
    static void generate_castling_moves(const Board& board, std::vector<Move>& moves);
    
    static bool is_attacked(const Board& board, int square, Color by_color);
};

} // namespace chess
