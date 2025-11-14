cmake_minimum_required(VERSION 3.15)

# Engine library
add_library(chess_engine STATIC
    src/board.cpp
    src/movegen.cpp
    src/search.cpp
    src/time_manager.cpp
    src/repetition.cpp
    src/nn_infer.cpp
    src/uci.cpp
)

target_include_directories(chess_engine PUBLIC include)

# Main executable
add_executable(chesshacks_engine src/main.cpp)
target_link_libraries(chesshacks_engine chess_engine)

# Tests
enable_testing()

add_executable(perft_tests tests/perft_tests.cpp)
target_link_libraries(perft_tests chess_engine)
add_test(NAME perft_tests COMMAND perft_tests)

add_executable(legality_tests tests/legality_tests.cpp)
target_link_libraries(legality_tests chess_engine)
add_test(NAME legality_tests COMMAND legality_tests)

add_executable(search_tests tests/search_tests.cpp)
target_link_libraries(search_tests chess_engine)
add_test(NAME search_tests COMMAND search_tests)
