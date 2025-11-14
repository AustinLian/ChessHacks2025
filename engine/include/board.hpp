#pragma once

#include <cstdint>
#include <array>
#include <string>

namespace chess {

// Piece types
enum PieceType : uint8_t {
    PAWN = 0,
    KNIGHT = 1,
    BISHOP = 2,
    ROOK = 3,
    QUEEN = 4,
    KING = 5,
    NO_PIECE = 6
};

// Colors
enum Color : uint8_t {
    WHITE = 0,
    BLACK = 1,
    NO_COLOR = 2
};

// Castling rights
enum CastlingRights : uint8_t {
    NO_CASTLING = 0,
    WHITE_OO = 1,
    WHITE_OOO = 2,
    BLACK_OO = 4,
    BLACK_OOO = 8,
    ALL_CASTLING = 15
};

// Board representation
class Board {
public:
    Board();
    explicit Board(const std::string& fen);
    
    // Board state
    void reset();
    bool set_fen(const std::string& fen);
    std::string to_fen() const;
    
    // Accessors
    PieceType piece_at(int square) const;
    Color color_at(int square) const;
    Color side_to_move() const;
    uint8_t castling_rights() const;
    int en_passant_square() const;
    int halfmove_clock() const;
    int fullmove_number() const;
    
    // Hash for transposition table
    uint64_t hash() const;
    
private:
    std::array<PieceType, 64> pieces_;
    std::array<Color, 64> colors_;
    Color side_to_move_;
    uint8_t castling_rights_;
    int en_passant_square_;
    int halfmove_clock_;
    int fullmove_number_;
    uint64_t hash_;
    
    void update_hash();
};

} // namespace chess
