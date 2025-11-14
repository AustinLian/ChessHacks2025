#pragma once

#include <cstdint>
#include <string>

namespace chess {

// Move representation (16-bit)
// bits 0-5: from square (0-63)
// bits 6-11: to square (0-63)
// bits 12-13: promotion piece type (0=none, 1=knight, 2=bishop, 3=rook, 4=queen)
// bits 14-15: special move flags (0=normal, 1=promotion, 2=en passant, 3=castling)

class Move {
public:
    Move() : data_(0) {}
    Move(int from, int to, int promotion = 0, int flags = 0);
    
    int from() const { return data_ & 0x3F; }
    int to() const { return (data_ >> 6) & 0x3F; }
    int promotion() const { return (data_ >> 12) & 0x3; }
    int flags() const { return (data_ >> 14) & 0x3; }
    
    bool is_promotion() const { return (data_ >> 14) == 1; }
    bool is_en_passant() const { return (data_ >> 14) == 2; }
    bool is_castling() const { return (data_ >> 14) == 3; }
    
    uint16_t data() const { return data_; }
    
    std::string to_uci() const;
    static Move from_uci(const std::string& uci);
    
    bool operator==(const Move& other) const { return data_ == other.data_; }
    bool operator!=(const Move& other) const { return data_ != other.data_; }
    
private:
    uint16_t data_;
};

} // namespace chess
