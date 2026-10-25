#pragma once
#include <stdexcept>
#include <string>

class DimensionMismatchError : public std::invalid_argument {
public:
    explicit DimensionMismatchError(const std::string& message)
        : std::invalid_argument(message) {}
};

class InvalidModelArgumentError : public std::invalid_argument {
public:
    explicit InvalidModelArgumentError(const std::string& message) : std::invalid_argument(message) {}
};

class ModelNotFittedError : public std::runtime_error {
public:
    explicit ModelNotFittedError(const std::string& message) : std::runtime_error(message) {}
};

class OutOfRangeError : public std::out_of_range {
public:
    explicit OutOfRangeError(const std::string& message) : std::out_of_range(message) {}
};