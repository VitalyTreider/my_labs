/**
 * @file exceptions.hpp
 * @brief Иерархия пользовательских исключений ядра библиотеки.
 */
#pragma once
#include <stdexcept>
#include <string>

/**
 * @brief Исключение, выбрасываемое при несовпадении размерностей матриц и векторов.
 *
 * Выбрасывается, если число строк матрицы признаков не совпадает с длиной вектора
 * откликов при обучении, либо если число столбцов при предсказании не равно числу
 * признаков, на котором обучалась модель.
 */
class DimensionMismatchError : public std::invalid_argument {
public:
    explicit DimensionMismatchError(const std::string& message)
        : std::invalid_argument(message) {}
};

/**
 * @brief Исключение, выбрасываемое при передаче некорректных аргументов или гиперпараметров.
 *
 * Выбрасывается при `min_samples_leaf < 1`, пустой матрице признаков, некорректном
 * `train_ratio` в `Dataset::split` или отсутствии целевого столбца.
 */
class InvalidModelArgumentError : public std::invalid_argument {
public:
    explicit InvalidModelArgumentError(const std::string& message)
        : std::invalid_argument(message) {}
};

/**
 * @brief Исключение, выбрасываемое при обращении к необученной модели.
 *
 * Выбрасывается при вызове `predict()` или геттеров параметров разбиения до вызова `fit()`.
 */
class ModelNotFittedError : public std::runtime_error {
public:
    explicit ModelNotFittedError(const std::string& message)
        : std::runtime_error(message) {}
};

/**
 * @brief Исключение, выбрасываемое при выходе индексов за границы матрицы в `Matrix::at()`.
 */
class OutOfRangeError : public std::out_of_range {
public:
    explicit OutOfRangeError(const std::string& message)
        : std::out_of_range(message) {}
};