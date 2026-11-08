#include "decision_stump.hpp"
#include "exceptions.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <unordered_map>


DecisionStump::DecisionStump(SplitCriterion criterion, std::size_t min_samples_leaf)
    : criterion_(criterion), min_samples_leaf_(min_samples_leaf) 
{
    if (min_samples_leaf < 1) {
        throw InvalidModelArgumentError("min_samples_leaf must be at least 1");
    }
}

double DecisionStump::left_value() const {
    if (!is_fitted_) {
        throw ModelNotFittedError("Model is not fitted yet");
    } else {
        return left_value_;
    }
}

double DecisionStump::right_value() const {
    if (!is_fitted_) {
        throw ModelNotFittedError("Model is not fitted yet");
    } else {
        return right_value_;
    }
}

bool DecisionStump::has_split() const {
    if (!is_fitted_) {
        throw ModelNotFittedError("Model is not fitted yet");
    } else {
        return has_split_;
    }
}

std::size_t DecisionStump::split_feature() const {
    if (!is_fitted_) {
        throw ModelNotFittedError("Model is not fitted yet");
    } else {
        return split_feature_;
    }
}

double DecisionStump::split_threshold() const {
    if (!is_fitted_) {
        throw ModelNotFittedError("Model is not fitted yet");
    } else {
        return split_threshold_;
    }
}

std::vector<double> DecisionStump::predict(const Matrix& features) const {
    if (!is_fitted_) {
        throw ModelNotFittedError("Model is not fitted yet");
    }

    if (features.cols() != n_features_) {
        throw DimensionMismatchError("Feature dimension mismatch in prediction");
    }

    std::vector<double> preds(features.rows());

    for (std::size_t i = 0; i < features.rows();++i) {
        if (!has_split_) {
            preds[i] = left_value_;
        } else {
            if (features[i][split_feature_] <= split_threshold_) {
                preds[i] = left_value_;
            } else {
                preds[i] = right_value_;
            }
        }
    }
    return preds;
}

void DecisionStump::fit(const Matrix& features, std::span<const double> target) {
    if (features.rows() == 0 || features.cols() == 0) {
        throw InvalidModelArgumentError("Features matrix is empty");
    }
    if (features.rows() != target.size()) {
        throw DimensionMismatchError("Feature and target row count mismatch");
    }

    //сброс модели перед обучением
    const std::size_t n = features.rows();
    const std::size_t d = features.cols();
    n_features_ = d;
    has_split_ = false;
    split_feature_ = 0;
    split_threshold_ = 0.0;

    //подготовка агрегаторов
    double total_sum = 0.0;

    // перенумерация для классификации
    std::unordered_map<double, std::size_t> label_to_id;
    std::vector<double> id_to_label;
    std::vector<std::size_t> encoded_target;
    std::vector<std::size_t> total_class_counts;
    double total_A = 0.0; 

    if (criterion_ == SplitCriterion::Variance) {
        total_sum = std::accumulate(target.begin(), target.end(), 0.0);
    } else {
        encoded_target.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            double lbl = target[i];
            auto it = label_to_id.find(lbl);
            if (it == label_to_id.end()) {
                std::size_t new_id = id_to_label.size();
                label_to_id[lbl] = new_id;
                id_to_label.push_back(lbl);
                total_class_counts.push_back(0);
                encoded_target[i] = new_id;
                total_class_counts[new_id]++;
            } else {
                encoded_target[i] = it->second;
                total_class_counts[it->second]++;
            }
        }
        for (std::size_t count : total_class_counts) {
            double c = static_cast<double>(count);
            total_A += c * c;
        }
    }

    std::vector<std::size_t> order(n);
    std::vector<std::size_t> left_counts(id_to_label.size(), 0);

    double best_score = -std::numeric_limits<double>::infinity();

    //цикл по признакам: O(d*nlogn)
    if (n >= 2) {
        for (std::size_t j = 0; j < d; ++j) {
            //заполняем order числами от 0 до n-1
            std::iota(order.begin(), order.end(), 0);

            //сортируем индексы лямбдой
            std::ranges::sort(order, [&features, j](std::size_t a, std::size_t b) {
                return features[a][j] < features[b][j];
            });

            
            std::size_t m_left = 0;
            std::size_t m_right = n;

            double sum_left = 0.0;
            double A_left = 0.0;
            double A_right = total_A;

            if (criterion_ == SplitCriterion::Gini) {
                std::ranges::fill(left_counts, 0);
            }

            //цикл кандидатов в пороги
            for (std::size_t p = 0; p <= n - 2; ++p) {
                const std::size_t i = order[p];

                
                m_left += 1;
                m_right -= 1;

                if (criterion_ == SplitCriterion::Variance) {
                    sum_left += target[i];
                } else {
                    const std::size_t c = encoded_target[i];
                    const double a = static_cast<double>(left_counts[c]);
                    const double b = static_cast<double>(total_class_counts[c] - left_counts[c]);

                    A_left += 2.0 * a + 1.0;
                    A_right += 1.0 - 2.0 * b;

                    left_counts[c] += 1;
                }

            
                const double val_curr = features[order[p]][j];
                const double val_next = features[order[p + 1]][j];
                if (val_curr == val_next) {
                    continue;
                }

                // огранич на мин размер листа
                if (m_left < min_samples_leaf_ || m_right < min_samples_leaf_) {
                    continue;
                }

                
                double score = 0.0;
                if (criterion_ == SplitCriterion::Variance) {
                    const double sum_right = total_sum - sum_left;
                    score = (sum_left * sum_left) / static_cast<double>(m_left)
                          + (sum_right * sum_right) / static_cast<double>(m_right);
                } else {
                    score = (A_left / static_cast<double>(m_left))
                          + (A_right / static_cast<double>(m_right));
                }

                const double threshold = std::midpoint(val_curr, val_next); // порог

                if (score > best_score) {
                    best_score = score;
                    split_feature_ = j;
                    split_threshold_ = threshold;
                    has_split_ = true;
                }
            }
        }
    }

    // вычисление предсказаний для листьев
    auto compute_leaf_prediction = [&](const std::vector<std::size_t>& indices) -> double {
        if (criterion_ == SplitCriterion::Variance) {
            double sum = 0.0;
            for (std::size_t idx : indices) {
                sum += target[idx];
            }
            return sum / static_cast<double>(indices.size());
        } else {
            std::vector<std::size_t> counts(id_to_label.size(), 0);
            for (std::size_t idx : indices) {
                counts[encoded_target[idx]]++;
            }
            std::size_t best_cls = 0;
            for (std::size_t c = 1; c < counts.size(); ++c) {
                if (counts[c] > counts[best_cls] ||
                    (counts[c] == counts[best_cls] && id_to_label[c] < id_to_label[best_cls])) {
                    best_cls = c;
                }
            }
            return id_to_label[best_cls];
        }
    };

    if (!has_split_) {
        std::vector<std::size_t> all_indices(n);
        std::iota(all_indices.begin(), all_indices.end(), 0);
        left_value_ = compute_leaf_prediction(all_indices);
        right_value_ = left_value_;
    } else {
        std::vector<std::size_t> left_indices;
        std::vector<std::size_t> right_indices;
        for (std::size_t i = 0; i < n; ++i) {
            if (features[i][split_feature_] <= split_threshold_) {
                left_indices.push_back(i);
            } else {
                right_indices.push_back(i);
            }
        }
        left_value_ = compute_leaf_prediction(left_indices);
        right_value_ = compute_leaf_prediction(right_indices);
    }

    is_fitted_ = true;
}