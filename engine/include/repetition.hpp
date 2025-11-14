#pragma once

#include "board.hpp"
#include <vector>

namespace chess {

// Repetition detection
class RepetitionTracker {
public:
    RepetitionTracker();
    
    void push(uint64_t hash);
    void pop();
    void clear();
    
    bool is_repetition(uint64_t hash, int threshold = 3) const;
    int count_occurrences(uint64_t hash) const;
    
private:
    std::vector<uint64_t> history_;
};

} // namespace chess
