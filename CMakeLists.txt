# -----------------------------------------------------------------------------
# Лабораторная работа №1 — датасет Credit Approval (credit_approval.csv)
#
# Корневой CMake-файл. Три цели:
#   * lab_core  — статическая библиотека ядра (заголовки в include/,
#                 исходники в src/ — код студента);
#   * lab1_app  — консольное приложение (src/main.cpp), линкуется с lab_core;
#   * lab_tests — тестовое приложение (tests/test_main.cpp), зарегистрировано
#                 в CTest через enable_testing() + add_test().
#
# Сборка (Debug + санитайзеры ASan/UBSan):
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
#   cmake --build build
#   ctest  --test-dir build
# -----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.24)

project(lab1_credit_approval
        VERSION 1.0.0
        DESCRIPTION "Lab 1: предобработка датасета Credit Approval (C++20)"
        LANGUAGES CXX)

# --- Стандарт языка ----------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Тип сборки по умолчанию для single-config генераторов (Makefiles, Ninja).
# Если пользователь явно передал CMAKE_BUILD_TYPE, не переопределяем.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the build type." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE
                 PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# --- Предупреждения компилятора (все типы сборки) ----------------------------
add_compile_options(-Wall -Wextra)

# --- Санитайзеры: AddressSanitizer + UndefinedBehaviorSanitizer (только Debug) --
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "lab1: Debug build -> -fsanitize=address,undefined enabled")
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options   (-fsanitize=address,undefined -fno-omit-frame-pointer)
endif()

# --- Ядро: статическая библиотека lab_core -----------------------------------
# Заголовки: include/, исходники: src/. Здесь будет жить код студента.
add_library(lab_core STATIC
    src/dataset.cpp
)
target_include_directories(lab_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# --- Консольное приложение lab1_app ------------------------------------------
add_executable(lab1_app
    src/main.cpp
)
target_link_libraries(lab1_app PRIVATE lab_core)

# --- Тестовое приложение lab_tests (CTest) -----------------------------------
enable_testing()
add_executable(lab_tests
    tests/test_main.cpp
)
target_link_libraries(lab_tests PRIVATE lab_core)
add_test(NAME lab_tests COMMAND lab_tests)
