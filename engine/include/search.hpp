#pragma once

#include "board.hpp"
#include "move.hpp"
#include <vector>
#include <memory>

namespace chess {

class NNInference;

// Search result
struct SearchResult {
    Move best_move;
    float score;
    int depth;
    uint64_t nodes;
    std::vector<Move> principal_variation;
};

// Search engine
class Search {
public:
    explicit Search(std::shared_ptr<NNInference> nn);
    
    SearchResult search(Board& board, int max_depth, int64_t time_limit_ms);
    void stop();
    
    // Configuration
    void set_transposition_table_size(size_t mb);
    void enable_history_heuristic(bool enable);
    void enable_killer_moves(bool enable);
    
private:
    float alpha_beta(Board& board, int depth, float alpha, float beta, bool is_max);
    std::vector<Move> order_moves(const Board& board, const std::vector<Move>& moves);
    
    std::shared_ptr<NNInference> nn_;
    bool should_stop_;
    uint64_t nodes_searched_;
    
    // Heuristics
    bool use_history_;
    bool use_killers_;
};

} // namespace chess
