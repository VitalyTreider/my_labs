/**
 * @file decision_stump.hpp
 * @brief Реализация модели «Решающий пень» (дерево решений глубины 1).
 */
#pragma once

#include <cstddef>
#include <span>
#include <vector>
#include "model.hpp"

/**
 * @brief Критерий качества разбиения решающего пня.
 */
enum class SplitCriterion {
    Variance, ///< Минимизация взвешенной дисперсии (SSE) для задачи регрессии.
    Gini      ///< Минимизация взвешенного индекса Джини для задачи классификации.
};

/**
 * @brief Модель решающего пня (`DecisionStump`), разделяющая выборку по одному признаку и порогу.
 *
 * Поиск оптимального порога выполняется за время O(d * n log n) за счёт однократной
 * сортировки индексов на каждый признак и инкрементального пересчёта агрегатов за O(1).
 */
class DecisionStump : public IModel
{
private:
    SplitCriterion criterion_;
    std::size_t min_samples_leaf_;

    bool is_fitted_ = false;
    bool has_split_ = false;
    std::size_t n_features_ = 0;
    std::size_t split_feature_ = 0;
    double split_threshold_ = 0.0;
    double left_value_ = 0.0;
    double right_value_ = 0.0;

public:
    /**
     * @brief Конструирует решающий пень с заданным критерием и ограничением на размер листа.
     * @param criterion Критерий разбиения (`SplitCriterion::Variance` или `SplitCriterion::Gini`).
     * @param min_samples_leaf Минимально допустимое число объектов в каждом листе (>= 1).
     * @pre `min_samples_leaf >= 1`
     * @throws InvalidModelArgumentError Если `min_samples_leaf < 1`.
     */
    DecisionStump(SplitCriterion criterion, std::size_t min_samples_leaf = 1);

    /**
     * @copydoc IModel::fit
     */
    void fit(const Matrix& features, std::span<const double> target) override;

    /**
     * @copydoc IModel::predict
     */
    std::vector<double> predict(const Matrix& features) const override;

    /**
     * @brief Возвращает индекс признака, по которому выполнено разбиение.
     * @throws ModelNotFittedError Если модель не обучена.
     */
    std::size_t split_feature() const;

    /**
     * @brief Возвращает численное значение порога разбиения t.
     * @throws ModelNotFittedError Если модель не обучена.
     */
    double split_threshold() const;

    /**
     * @brief Возвращает предсказание в левом листе (для объектов с X[i][j] <= t).
     * @throws ModelNotFittedError Если модель не обучена.
     */
    double left_value() const;

    /**
     * @brief Возвращает предсказание в правом листе (для объектов с X[i][j] > t).
     * @throws ModelNotFittedError Если модель не обучена.
     */
    double right_value() const;

    /**
     * @brief Показывает, было ли найдено хотя бы одно допустимое разбиение.
     * @throws ModelNotFittedError Если модель не обучена.
     */
    bool has_split() const;
};