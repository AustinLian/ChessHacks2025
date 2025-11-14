cmake_minimum_required(VERSION 3.15)
project(ChessHacksEngine VERSION 0.1.0 LANGUAGES CXX)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# Add subdirectory for engine
add_subdirectory(engine)
