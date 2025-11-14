#include "uci.hpp"
#include "movegen.hpp"
#include "nn_infer.hpp"
#include <iostream>
#include <sstream>

namespace chess {

UCI::UCI() : should_quit_(false) {
    board_.reset();
    auto nn = std::make_shared<NNInference>();
    search_ = std::make_unique<Search>(nn);
}

UCI::~UCI() = default;

void UCI::run() {
    std::string line;
    while (!should_quit_ && std::getline(std::cin, line)) {
        handle_command(line);
    }
}

void UCI::stop() {
    should_quit_ = true;
}

void UCI::handle_command(const std::string& line) {
    std::istringstream iss(line);
    std::string cmd;
    iss >> cmd;
    
    if (cmd == "uci") {
        cmd_uci();
    } else if (cmd == "isready") {
        cmd_isready();
    } else if (cmd == "ucinewgame") {
        cmd_ucinewgame();
    } else if (cmd == "position") {
        std::string args;
        std::getline(iss, args);
        cmd_position(args);
    } else if (cmd == "go") {
        std::string args;
        std::getline(iss, args);
        cmd_go(args);
    } else if (cmd == "stop") {
        cmd_stop();
    } else if (cmd == "quit") {
        cmd_quit();
    }
}

void UCI::cmd_uci() {
    std::cout << "id name ChessHacks Engine 0.1.0" << std::endl;
    std::cout << "id author ChessHacks2025 Team" << std::endl;
    std::cout << "uciok" << std::endl;
}

void UCI::cmd_isready() {
    std::cout << "readyok" << std::endl;
}

void UCI::cmd_ucinewgame() {
    board_.reset();
}

void UCI::cmd_position(const std::string& args) {
    // TODO: Parse position command (fen or startpos + moves)
}

void UCI::cmd_go(const std::string& args) {
    // TODO: Parse go command (time controls, depth, nodes)
    auto result = search_->search(board_, 10, 5000);
    std::cout << "bestmove " << result.best_move.to_uci() << std::endl;
}

void UCI::cmd_stop() {
    search_->stop();
}

void UCI::cmd_quit() {
    should_quit_ = true;
}

void UCI::cmd_setoption(const std::string& args) {
    // TODO: Parse and set engine options
}

} // namespace chess
