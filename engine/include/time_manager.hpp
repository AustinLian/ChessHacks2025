#pragma once

#include <cstdint>

namespace chess {

// Time management for games
class TimeManager {
public:
    TimeManager();
    
    void set_time_control(int64_t base_time_ms, int64_t increment_ms);
    void set_move_overhead(int64_t overhead_ms);
    
    // Calculate time to allocate for current move
    int64_t allocate_time(int moves_played, int64_t time_remaining_ms) const;
    
    // Update after move
    void update_time_used(int64_t time_used_ms);
    
private:
    int64_t base_time_ms_;
    int64_t increment_ms_;
    int64_t move_overhead_ms_;
    float time_buffer_ratio_;
};

} // namespace chess
