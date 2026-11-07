#pragma once

#include <cstddef>
#include <span>
#include <vector>
#include "model.hpp"

enum class SplitCriterion { Variance, Gini };

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
    DecisionStump(SplitCriterion criterion, std::size_t min_samples_leaf = 1);

    void fit(const Matrix& features, std::span<const double> target) override;
    std::vector<double> predict(const Matrix& features) const override;

    std::size_t split_feature() const;
    double split_threshold() const;
    double left_value() const;
    double right_value() const;
    bool has_split() const;
};