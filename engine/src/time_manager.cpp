#include "time_manager.hpp"
#include <algorithm>

namespace chess {

TimeManager::TimeManager()
    : base_time_ms_(0), increment_ms_(0), move_overhead_ms_(50),
      time_buffer_ratio_(0.05f) {
}

void TimeManager::set_time_control(int64_t base_time_ms, int64_t increment_ms) {
    base_time_ms_ = base_time_ms;
    increment_ms_ = increment_ms;
}

void TimeManager::set_move_overhead(int64_t overhead_ms) {
    move_overhead_ms_ = overhead_ms;
}

int64_t TimeManager::allocate_time(int moves_played, int64_t time_remaining_ms) const {
    // Simple time allocation: divide remaining time by expected moves
    const int expected_moves = std::max(20, 40 - moves_played);
    
    int64_t allocated = (time_remaining_ms - move_overhead_ms_) / expected_moves;
    allocated += increment_ms_;
    
    // Apply safety buffer
    allocated = static_cast<int64_t>(allocated * (1.0f - time_buffer_ratio_));
    
    return std::max(int64_t(100), allocated);
}

void TimeManager::update_time_used(int64_t time_used_ms) {
    // TODO: Track time usage patterns for adaptive allocation
}

} // namespace chess
