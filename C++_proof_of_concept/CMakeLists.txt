cmake_minimum_required(VERSION 3.10)
project(MultibodyPhysics)

# Force maximum optimization and disable Eigen debug checks
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-O3 -DNDEBUG -march=native")

# Find Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Include custom headers
include_directories(${PROJECT_SOURCE_DIR}/include)

# --- Build the Open Loop target ---
add_executable(sim_open src/main_open.cpp)
target_link_libraries(sim_open Eigen3::Eigen)

# --- Build the Closed Loop target ---
add_executable(sim_closed src/main_closed.cpp)
target_link_libraries(sim_closed Eigen3::Eigen)