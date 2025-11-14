#include "board.hpp"
#include <sstream>
#include <algorithm>

namespace chess {

Board::Board() {
    reset();
}

Board::Board(const std::string& fen) {
    set_fen(fen);
}

void Board::reset() {
    // Set starting position
    set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

bool Board::set_fen(const std::string& fen) {
    // TODO: Implement FEN parsing
    // Parse piece placement, side to move, castling rights, en passant, clocks
    return true;
}

std::string Board::to_fen() const {
    // TODO: Implement FEN generation
    return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
}

PieceType Board::piece_at(int square) const {
    return pieces_[square];
}

Color Board::color_at(int square) const {
    return colors_[square];
}

Color Board::side_to_move() const {
    return side_to_move_;
}

uint8_t Board::castling_rights() const {
    return castling_rights_;
}

int Board::en_passant_square() const {
    return en_passant_square_;
}

int Board::halfmove_clock() const {
    return halfmove_clock_;
}

int Board::fullmove_number() const {
    return fullmove_number_;
}

uint64_t Board::hash() const {
    return hash_;
}

void Board::update_hash() {
    // TODO: Implement Zobrist hashing
    hash_ = 0;
}

} // namespace chess
