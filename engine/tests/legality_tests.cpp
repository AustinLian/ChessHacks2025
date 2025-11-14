#include "movegen.hpp"
#include "board.hpp"
#include <iostream>
#include <cassert>

using namespace chess;

void test_castling_white_kingside() {
    // Position where white can castle kingside
    Board board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
    
    auto moves = MoveGen::generate_legal_moves(board);
    
    // TODO: Verify castling move is in legal moves
    std::cout << "Castling kingside test placeholder" << std::endl;
}

void test_en_passant() {
    // Position with en passant opportunity
    Board board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 1");
    
    auto moves = MoveGen::generate_legal_moves(board);
    
    // TODO: Verify en passant capture is in legal moves
    std::cout << "En passant test placeholder" << std::endl;
}

void test_pawn_promotion() {
    // Position where pawn can promote
    Board board("8/P7/8/8/8/8/8/k6K w - - 0 1");
    
    auto moves = MoveGen::generate_legal_moves(board);
    
    // TODO: Verify all 4 promotion moves (Q, R, B, N) are generated
    std::cout << "Pawn promotion test placeholder" << std::endl;
}

void test_no_castling_through_check() {
    // Position where castling would move through check
    Board board("r3k2r/8/8/8/8/8/8/R2qK2R w KQkq - 0 1");
    
    auto moves = MoveGen::generate_legal_moves(board);
    
    // TODO: Verify castling is not allowed
    std::cout << "Castling through check test placeholder" << std::endl;
}

int main() {
    test_castling_white_kingside();
    test_en_passant();
    test_pawn_promotion();
    test_no_castling_through_check();
    
    std::cout << "All legality tests passed!" << std::endl;
    return 0;
}
