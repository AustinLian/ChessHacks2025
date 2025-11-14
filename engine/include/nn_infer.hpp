#pragma once

#include "board.hpp"
#include "move.hpp"
#include <string>
#include <vector>
#include <memory>

namespace chess {

// Neural network inference interface
class NNInference {
public:
    NNInference();
    ~NNInference();
    
    // Load network weights from file
    bool load_weights(const std::string& weights_path);
    
    // Evaluate position
    // Returns: (policy vector, value scalar)
    // policy: move probabilities (one per legal move)
    // value: expected outcome from side-to-move perspective (-1 to +1)
    struct EvalResult {
        std::vector<float> policy;  // policy[i] = probability of moves[i]
        float value;                 // -1 (loss) to +1 (win)
    };
    
    EvalResult evaluate(const Board& board, const std::vector<Move>& legal_moves);
    
    // Batch evaluation (for future parallelization)
    std::vector<EvalResult> evaluate_batch(
        const std::vector<Board>& boards,
        const std::vector<std::vector<Move>>& legal_moves);
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    // Convert board to neural network input planes
    void board_to_planes(const Board& board, float* planes) const;
};

} // namespace chess
