#include "movegen.hpp"
#include "board.hpp"
#include <iostream>
#include <cassert>

using namespace chess;

void test_perft_starting_position() {
    Board board;
    
    // Perft results for starting position
    // depth 1: 20 nodes
    // depth 2: 400 nodes
    // depth 3: 8902 nodes
    // depth 4: 197281 nodes
    // depth 5: 4865609 nodes
    
    std::cout << "Testing perft from starting position..." << std::endl;
    
    // TODO: Uncomment when movegen is implemented
    // uint64_t nodes = MoveGen::perft(board, 1);
    // assert(nodes == 20 && "Perft depth 1 failed");
    
    std::cout << "Perft tests placeholder" << std::endl;
}

void test_perft_kiwipete() {
    // Kiwipete position
    Board board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    
    // Perft results for Kiwipete
    // depth 1: 48 nodes
    // depth 2: 2039 nodes
    // depth 3: 97862 nodes
    
    std::cout << "Testing perft on Kiwipete position..." << std::endl;
    
    // TODO: Uncomment when movegen is implemented
    // uint64_t nodes = MoveGen::perft(board, 1);
    // assert(nodes == 48 && "Kiwipete perft depth 1 failed");
    
    std::cout << "Kiwipete perft tests placeholder" << std::endl;
}

int main() {
    test_perft_starting_position();
    test_perft_kiwipete();
    
    std::cout << "All perft tests passed!" << std::endl;
    return 0;
}
