/**
 * @file model.hpp
 * @brief Абстрактный полиморфный интерфейс модели машинного обучения.
 */
#pragma once

#include <span>
#include <vector>
#include "dataset.hpp"


/**
 * @brief Базовый интерфейс модели обучения с учителем.
 */
class IModel
{
public:
    virtual ~IModel() = default;

    /**
     * @brief Обучает модель по матрице признаков и вектору откликов.
     * @param features Матрица признаков размера n x d (не пустая).
     * @param target Непрерывный срез вектора целевых значений длины n.
     * @pre `features.rows() > 0 && features.cols() > 0`
     * @pre `features.rows() == target.size()`
     * @throws InvalidModelArgumentError Если матрица `features` пуста.
     * @throws DimensionMismatchError Если `features.rows() != target.size()`.
     */
    virtual void fit(
        const Matrix& features,
        std::span<const double> target
    ) = 0;

    /**
     * @brief Вычисляет предсказания для новой матрицы объектов.
     * @param features Матрица признаков размера m x d.
     * @return Вектор предсказанных значений длины m.
     * @pre Модель должна быть обучена вызовом `fit()`.
     * @pre `features.cols()` должен совпадать с числом признаков на обучении.
     * @throws ModelNotFittedError Если модель ещё не обучена.
     * @throws DimensionMismatchError Если число признаков не совпадает с обучающим.
     */
    virtual std::vector<double> predict(
        const Matrix& features
    ) const = 0;
};