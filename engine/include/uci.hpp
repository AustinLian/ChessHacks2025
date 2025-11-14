#pragma once

#include "board.hpp"
#include "search.hpp"
#include <string>
#include <memory>

namespace chess {

// UCI / Simple CLI protocol handler
class UCI {
public:
    UCI();
    ~UCI();
    
    void run();
    void stop();
    
private:
    void handle_command(const std::string& line);
    
    // UCI commands
    void cmd_uci();
    void cmd_isready();
    void cmd_ucinewgame();
    void cmd_position(const std::string& args);
    void cmd_go(const std::string& args);
    void cmd_stop();
    void cmd_quit();
    
    // Engine options
    void cmd_setoption(const std::string& args);
    
    Board board_;
    std::unique_ptr<Search> search_;
    bool should_quit_;
};

} // namespace chess
