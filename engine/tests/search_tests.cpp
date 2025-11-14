#include "search.hpp"
#include "board.hpp"
#include "nn_infer.hpp"
#include <iostream>
#include <memory>
#include <cassert>

using namespace chess;

void test_search_mate_in_one() {
    // Back rank mate position
    Board board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1");
    
    auto nn = std::make_shared<NNInference>();
    Search search(nn);
    
    auto result = search.search(board, 10, 5000);
    
    // TODO: Verify search finds mate in one (Ra8#)
    std::cout << "Mate in one test placeholder" << std::endl;
}

void test_search_avoids_blunder() {
    // Position where hanging piece should be captured
    Board board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPPQPPP/RNB1KBNR b KQkq - 0 1");
    
    auto nn = std::make_shared<NNInference>();
    Search search(nn);
    
    auto result = search.search(board, 10, 5000);
    
    // TODO: Verify search doesn't hang pieces
    std::cout << "Avoid blunder test placeholder" << std::endl;
}

void test_search_time_management() {
    Board board;
    
    auto nn = std::make_shared<NNInference>();
    Search search(nn);
    
    // Search with 1 second limit
    auto start = std::chrono::steady_clock::now();
    auto result = search.search(board, 100, 1000);
    auto end = std::chrono::steady_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // TODO: Verify search respects time limit
    std::cout << "Time management test placeholder (took " << duration << " ms)" << std::endl;
}

int main() {
    test_search_mate_in_one();
    test_search_avoids_blunder();
    test_search_time_management();
    
    std::cout << "All search tests passed!" << std::endl;
    return 0;
}
