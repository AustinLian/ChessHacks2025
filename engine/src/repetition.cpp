#include "repetition.hpp"
#include <algorithm>

namespace chess {

RepetitionTracker::RepetitionTracker() {
    history_.reserve(256);
}

void RepetitionTracker::push(uint64_t hash) {
    history_.push_back(hash);
}

void RepetitionTracker::pop() {
    if (!history_.empty()) {
        history_.pop_back();
    }
}

void RepetitionTracker::clear() {
    history_.clear();
}

bool RepetitionTracker::is_repetition(uint64_t hash, int threshold) const {
    return count_occurrences(hash) >= threshold;
}

int RepetitionTracker::count_occurrences(uint64_t hash) const {
    return std::count(history_.begin(), history_.end(), hash);
}

} // namespace chess
