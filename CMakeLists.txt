cmake_minimum_required(VERSION 3.24)

project(lab1_credit_approval
        VERSION 1.0.0
        DESCRIPTION "Labs 1-2: Credit Approval preprocessing and Decision Stump (C++20)"
        LANGUAGES CXX)

# --- Стандарт языка ----------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the build type." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE
                 PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# --- Предупреждения компилятора (все типы сборки) ----------------------------
add_compile_options(-Wall -Wextra)

# --- Санитайзеры: AddressSanitizer + UndefinedBehaviorSanitizer (только Debug) --
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug build -> -fsanitize=address,undefined enabled")
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options   (-fsanitize=address,undefined -fno-omit-frame-pointer)
endif()

# --- Ядро: статическая библиотека lab_core (БЕЗ std::cout и CLI) -------------
add_library(lab_core STATIC
    src/dataset.cpp
    src/decision_stump.cpp
)
target_include_directories(lab_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

include(FetchContent)
FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG        v3.2.0
)
FetchContent_MakeAvailable(cxxopts)

# --- Консольное приложение lab1_app ------------------------------------------
add_executable(lab1_app
    src/main.cpp
    src/view.cpp
)
target_link_libraries(lab1_app PRIVATE lab_core cxxopts::cxxopts)

# --- Консольное приложение lab2_app ------------------------------------------
add_executable(lab2_app
    src/lab2_main.cpp
)
target_link_libraries(lab2_app PRIVATE lab_core cxxopts::cxxopts)

# --- Тестовые приложения (CTest) ---------------------------------------------
enable_testing()

add_executable(lab_tests
    tests/test_main.cpp
)
target_link_libraries(lab_tests PRIVATE lab_core)
add_test(NAME lab_tests COMMAND lab_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(lab2_tests
    tests/lab2_tests.cpp
)
target_link_libraries(lab2_tests PRIVATE lab_core)
add_test(NAME lab2_tests COMMAND lab2_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})