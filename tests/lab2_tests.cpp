#include "dataset.hpp"
#include "decision_stump.hpp"
#include "exceptions.hpp"
#include "model.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <span>
#include <vector>

namespace {

constexpr double kEps = 1e-9;

bool approx_equal(double a, double b, double eps = kEps) {
    return std::abs(a - b) <= eps;
}

// наивная реализация решающего пня за O(d * n^2), живущая только в тестах
struct NaiveStumpResult {
    bool has_split = false;
    std::size_t split_feature = 0;
    double split_threshold = 0.0;
    double left_value = 0.0;
    double right_value = 0.0;
};

double naive_leaf_prediction(
    std::span<const double> y,
    const std::vector<std::size_t>& indices,
    SplitCriterion criterion
) {
    if (criterion == SplitCriterion::Variance) {
        double sum = 0.0;
        for (std::size_t idx : indices) {
            sum += y[idx];
        }
        return sum / static_cast<double>(indices.size());
    } else {
        std::map<double, std::size_t> counts;
        for (std::size_t idx : indices) {
            counts[y[idx]]++;
        }
        double best_label = counts.begin()->first;
        std::size_t best_count = counts.begin()->second;
        for (const auto& [label, cnt] : counts) {
            if (cnt > best_count) {
                best_count = cnt;
                best_label = label;
            }
        }
        return best_label;
    }
}

double naive_leaf_impurity_weighted(
    std::span<const double> y,
    const std::vector<std::size_t>& indices,
    SplitCriterion criterion
) {
    const double m = static_cast<double>(indices.size());
    if (criterion == SplitCriterion::Variance) {
        double sum = 0.0;
        for (std::size_t idx : indices) {
            sum += y[idx];
        }
        const double mean = sum / m;
        double sse = 0.0;
        for (std::size_t idx : indices) {
            const double diff = y[idx] - mean;
            sse += diff * diff;
        }
        return sse;
    } else {
        std::map<double, std::size_t> counts;
        for (std::size_t idx : indices) {
            counts[y[idx]]++;
        }
        double sum_sq_proportions = 0.0;
        for (const auto& [label, cnt] : counts) {
            const double p = static_cast<double>(cnt) / m;
            sum_sq_proportions += p * p;
        }
        const double gini = 1.0 - sum_sq_proportions;
        return m * gini;
    }
}

NaiveStumpResult fit_naive_stump(
    const Matrix& features,
    std::span<const double> target,
    SplitCriterion criterion,
    std::size_t min_samples_leaf
) {
    NaiveStumpResult res;
    const std::size_t n = features.rows();
    const std::size_t d = features.cols();

    double best_loss = std::numeric_limits<double>::infinity();

    for (std::size_t j = 0; j < d; ++j) {
        std::vector<double> col_vals(n);
        for (std::size_t i = 0; i < n; ++i) {
            col_vals[i] = features[i][j];
        }
        std::ranges::sort(col_vals);

        for (std::size_t p = 0; p + 1 < n; ++p) {
            if (col_vals[p] == col_vals[p + 1]) {
                continue;
            }
            const double threshold = std::midpoint(col_vals[p], col_vals[p + 1]);

            //полный проход по всей выборке O(n)
            std::vector<std::size_t> left_idx;
            std::vector<std::size_t> right_idx;
            for (std::size_t i = 0; i < n; ++i) {
                if (features[i][j] <= threshold) {
                    left_idx.push_back(i);
                } else {
                    right_idx.push_back(i);
                }
            }

            if (left_idx.size() < min_samples_leaf || right_idx.size() < min_samples_leaf) {
                continue;
            }

            const double loss =
                naive_leaf_impurity_weighted(target, left_idx, criterion) +
                naive_leaf_impurity_weighted(target, right_idx, criterion);

            if (loss < best_loss - 1e-12) {
                best_loss = loss;
                res.has_split = true;
                res.split_feature = j;
                res.split_threshold = threshold;
                res.left_value = naive_leaf_prediction(target, left_idx, criterion);
                res.right_value = naive_leaf_prediction(target, right_idx, criterion);
            }
        }
    }

    if (!res.has_split) {
        std::vector<std::size_t> all_idx(n);
        std::iota(all_idx.begin(), all_idx.end(), 0);
        res.left_value = naive_leaf_prediction(target, all_idx, criterion);
        res.right_value = res.left_value;
    }

    return res;
}

//сравнение с наивной реализацией на случайных данных
void test_compare_with_naive() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> feat_dist(-10.0, 10.0);
    std::uniform_real_distribution<double> reg_target_dist(-50.0, 50.0);
    std::uniform_int_distribution<int> cls_target_dist(0, 2);

    const std::size_t n = 60;
    const std::size_t d = 5;

    for (int trial = 0; trial < 5; ++trial) {
        Matrix X(n, d);
        std::vector<double> y_reg(n);
        std::vector<double> y_cls(n);

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < d; ++j) {
                double val = feat_dist(rng);
                if (j == 2) {
                    val = std::round(val);
                }
                X[i][j] = val;
            }
            y_reg[i] = reg_target_dist(rng);
            y_cls[i] = static_cast<double>(cls_target_dist(rng));
        }

        for (std::size_t min_leaf : {std::size_t{1}, std::size_t{3}, std::size_t{10}, std::size_t{25}}) {
            {
                DecisionStump fast(SplitCriterion::Variance, min_leaf);
                fast.fit(X, y_reg);
                NaiveStumpResult naive = fit_naive_stump(X, y_reg, SplitCriterion::Variance, min_leaf);

                assert(fast.has_split() == naive.has_split);
                if (fast.has_split()) {
                    assert(fast.split_feature() == naive.split_feature);
                    assert(approx_equal(fast.split_threshold(), naive.split_threshold));
                }
                assert(approx_equal(fast.left_value(), naive.left_value));
                assert(approx_equal(fast.right_value(), naive.right_value));
            }

            {
                DecisionStump fast(SplitCriterion::Gini, min_leaf);
                fast.fit(X, y_cls);
                NaiveStumpResult naive = fit_naive_stump(X, y_cls, SplitCriterion::Gini, min_leaf);

                assert(fast.has_split() == naive.has_split);
                if (fast.has_split()) {
                    assert(fast.split_feature() == naive.split_feature);
                    assert(approx_equal(fast.split_threshold(), naive.split_threshold));
                }
                assert(approx_equal(fast.left_value(), naive.left_value));
                assert(approx_equal(fast.right_value(), naive.right_value));
            }
        }
    }
    std::cout << "[OK] Test 1: Сравнение с наивной реализацией (Variance и Gini) пройдено\n";
}

//искусственные данные, разделимые по одному признаку
void test_separable_synthetic_data() {
    Matrix X(6, 2);
    X[0][0] = 5.0; X[0][1] = 1.0;
    X[1][0] = 1.0; X[1][1] = 2.0;
    X[2][0] = 9.0; X[2][1] = 1.5;
    X[3][0] = 2.0; X[3][1] = 8.0;
    X[4][0] = 8.0; X[4][1] = 9.0;
    X[5][0] = 4.0; X[5][1] = 8.5;

    std::vector<double> y_cls = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    DecisionStump stump_cls(SplitCriterion::Gini, 1);
    stump_cls.fit(X, y_cls);

    assert(stump_cls.has_split());
    assert(stump_cls.split_feature() == 1);
    assert(approx_equal(stump_cls.split_threshold(), 5.0));
    assert(approx_equal(stump_cls.left_value(), 0.0));
    assert(approx_equal(stump_cls.right_value(), 1.0));

    std::vector<double> y_reg = {10.0, 10.0, 10.0, 100.0, 100.0, 100.0};
    DecisionStump stump_reg(SplitCriterion::Variance, 1);
    stump_reg.fit(X, y_reg);

    assert(stump_reg.has_split());
    assert(stump_reg.split_feature() == 1);
    assert(approx_equal(stump_reg.split_threshold(), 5.0));
    assert(approx_equal(stump_reg.left_value(), 10.0));
    assert(approx_equal(stump_reg.right_value(), 100.0));

    std::cout << "[OK] Test 2: Искусственные разделимые данные пройдены\n";
}

//признак-константа не выбирается, если есть информативный
void test_constant_feature_ignored() {
    Matrix X(4, 2);
    X[0][0] = 7.0; X[0][1] = 1.0;
    X[1][0] = 7.0; X[1][1] = 2.0;
    X[2][0] = 7.0; X[2][1] = 10.0;
    X[3][0] = 7.0; X[3][1] = 12.0;

    std::vector<double> y = {0.0, 0.0, 1.0, 1.0};
    DecisionStump stump(SplitCriterion::Gini, 1);
    stump.fit(X, y);

    assert(stump.has_split());
    assert(stump.split_feature() == 1);
    assert(approx_equal(stump.split_threshold(), 6.0));

    Matrix X_const(4, 1);
    for (std::size_t i = 0; i < 4; ++i) {
        X_const[i][0] = 42.0;
    }
    DecisionStump stump_no_split(SplitCriterion::Variance, 1);
    stump_no_split.fit(X_const, y);
    assert(!stump_no_split.has_split());
    assert(approx_equal(stump_no_split.left_value(), 0.5));

    std::cout << "[OK] Test 3: Игнорирование признака-константы пройдено\n";
}

//порог не ставится между равными значениями
void test_duplicate_feature_values() {
    Matrix X(4, 1);
    X[0][0] = 1.0;
    X[1][0] = 1.0;
    X[2][0] = 1.0;
    X[3][0] = 5.0;

    std::vector<double> y = {0.0, 0.0, 1.0, 1.0};

    DecisionStump stump_leaf2(SplitCriterion::Gini, 2);
    stump_leaf2.fit(X, y);
    assert(!stump_leaf2.has_split());

    DecisionStump stump_leaf1(SplitCriterion::Gini, 1);
    stump_leaf1.fit(X, y);
    assert(stump_leaf1.has_split());
    assert(approx_equal(stump_leaf1.split_threshold(), 3.0));

    std::cout << "[OK] Test 4: Обработка дубликатов значений признака пройдена\n";
}

//пень подставляется через интерфейс IModel
std::vector<double> run_through_imodel(
    IModel& model,
    const Matrix& train_X,
    std::span<const double> train_y,
    const Matrix& test_X
) {
    model.fit(train_X, train_y);
    return model.predict(test_X);
}

void test_polymorphic_imodel() {
    Matrix train_X(4, 1);
    train_X[0][0] = 1.0;
    train_X[1][0] = 2.0;
    train_X[2][0] = 8.0;
    train_X[3][0] = 9.0;
    std::vector<double> train_y = {0.0, 0.0, 1.0, 1.0};

    Matrix test_X(2, 1);
    test_X[0][0] = 1.5;
    test_X[1][0] = 8.5;

    DecisionStump stump(SplitCriterion::Gini, 1);
    std::vector<double> preds = run_through_imodel(stump, train_X, train_y, test_X);

    assert(preds.size() == 2);
    assert(approx_equal(preds[0], 0.0));
    assert(approx_equal(preds[1], 1.0));

    std::cout << "[OK] Test 5: Полиморфная подстановка через IModel пройдена\n";
}

//все 6 обязательных исключений
void test_all_required_exceptions() {
    // 1. min_samples_leaf < 1 -> InvalidModelArgumentError
    {
        bool caught = false;
        try {
            DecisionStump bad(SplitCriterion::Gini, 0);
        } catch (const InvalidModelArgumentError&) {
            caught = true;
        }
        assert(caught);
    }

    // 2. predict вызван до fit -> ModelNotFittedError
    {
        DecisionStump unfitted(SplitCriterion::Variance, 1);
        Matrix X(2, 2);
        bool caught_pred = false;
        try {
            (void)unfitted.predict(X);
        } catch (const ModelNotFittedError&) {
            caught_pred = true;
        }
        assert(caught_pred);
    }

    // 3. обучение на пустой выборке -> InvalidModelArgumentError
    {
        DecisionStump stump(SplitCriterion::Gini, 1);
        Matrix empty_X(0, 0);
        std::vector<double> empty_y;
        bool caught = false;
        try {
            stump.fit(empty_X, empty_y);
        } catch (const InvalidModelArgumentError&) {
            caught = true;
        }
        assert(caught);
    }

    // 4. несовпадение числа объектов в features и длины target -> DimensionMismatchError
    {
        DecisionStump stump(SplitCriterion::Gini, 1);
        Matrix X(4, 2);
        std::vector<double> y = {0.0, 1.0};
        bool caught = false;
        try {
            stump.fit(X, y);
        } catch (const DimensionMismatchError&) {
            caught = true;
        }
        assert(caught);
    }

    // 5. при предсказании число признаков не совпадает с обучающим -> DimensionMismatchError
    {
        DecisionStump stump(SplitCriterion::Gini, 1);
        Matrix train_X(4, 2);
        std::vector<double> train_y = {0.0, 0.0, 1.0, 1.0};
        stump.fit(train_X, train_y);

        Matrix bad_test_X(2, 3);
        bool caught = false;
        try {
            (void)stump.predict(bad_test_X);
        } catch (const DimensionMismatchError&) {
            caught = true;
        }
        assert(caught);
    }

    // 6. выход за границы в Matrix::at() -> OutOfRangeError
    {
        Matrix m(2, 3);
        bool caught = false;
        try {
            (void)m.at(2, 0);
        } catch (const OutOfRangeError&) {
            caught = true;
        }
        assert(caught);
    }

    std::cout << "[OK] Test 6: Все 6 обязательных типов исключений проверены\n";
}

// доп бенчмарк на реальном датасете (если файл доступен)
void benchmark_on_real_dataset() {
    std::filesystem::path csv_path = "data/credit_approval.csv";
    if (!std::filesystem::exists(csv_path)) {
        csv_path = "../data/credit_approval.csv";
    }
    if (!std::filesystem::exists(csv_path)) {
        return;
    }

    CsvConfig cfg;
    Dataset full_ds = Dataset::load(csv_path, cfg);
    auto [train_ds, test_ds] = full_ds.split(0.8);
    DatasetInfo info = train_ds.analyze();
    train_ds.impute(info, NumericImputation::Median);
    TransformedDataset train_td = train_ds.transform(
        info, Scaling::MinMax, Layout::RowMajor, CategoricalEncoding::OneHot, "A16"
    );

    DecisionStump fast_stump(SplitCriterion::Gini, 5);
    const auto t0 = std::chrono::high_resolution_clock::now();
    fast_stump.fit(train_td.matrix, train_td.target);
    const auto t1 = std::chrono::high_resolution_clock::now();

    const auto t2 = std::chrono::high_resolution_clock::now();
    NaiveStumpResult naive_res = fit_naive_stump(train_td.matrix, train_td.target, SplitCriterion::Gini, 5);
    const auto t3 = std::chrono::high_resolution_clock::now();

    assert(fast_stump.split_feature() == naive_res.split_feature);
    assert(approx_equal(fast_stump.split_threshold(), naive_res.split_threshold));

    const double fast_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double naive_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "[BENCHMARK credit_approval.csv (552 x 51)] Быстрый O(d*n log n): " << fast_ms
              << " мс | Наивный O(d*n^2): " << naive_ms << " мс\n";
}

// доп тест всех граничных случаев из раздела 9
void test_edge_cases_section_9() {
    // 1) в выборке ровно один объект (n = 1) -> разбиения нет, предсказывается константа
    {
        Matrix X(1, 2);
        X[0][0] = 3.5;
        X[0][1] = -1.2;
        std::vector<double> y = {42.0};

        DecisionStump stump(SplitCriterion::Variance, 1);
        stump.fit(X, y);
        assert(!stump.has_split());
        assert(approx_equal(stump.left_value(), 42.0));
        auto preds = stump.predict(X);
        assert(preds.size() == 1 && approx_equal(preds[0], 42.0));
    }

    // 2) в выборке один класс (все y одинаковы) -> предсказывается этот единственный класс
    {
        Matrix X(4, 1);
        X[0][0] = 1.0; X[1][0] = 2.0; X[2][0] = 3.0; X[3][0] = 4.0;
        std::vector<double> y = {7.0, 7.0, 7.0, 7.0};

        DecisionStump stump(SplitCriterion::Gini, 1);
        stump.fit(X, y);
        auto preds = stump.predict(X);
        for (double p : preds) {
            assert(approx_equal(p, 7.0));
        }
    }

    // 3) min_samples_leaf слишком велик и не позволяет ни одного разбиения
    {
        Matrix X(4, 1);
        X[0][0] = 1.0; X[1][0] = 2.0; X[2][0] = 10.0; X[3][0] = 20.0;
        std::vector<double> y = {0.0, 0.0, 1.0, 1.0};

        DecisionStump stump(SplitCriterion::Gini, 3); // при n=4 нельзя получить >=3 и слева, и справа
        stump.fit(X, y);
        assert(!stump.has_split());
    }

    // 4)при равенстве критерия: выбирается меньший индекс признака, затем меньший порог
    {
        Matrix X(4, 2);
        // Признак 0 и Признак 1 оба идеально разделяют выборку с одинаковым качеством
        X[0][0] = 1.0; X[0][1] = 10.0;
        X[1][0] = 2.0; X[1][1] = 20.0;
        X[2][0] = 5.0; X[2][1] = 50.0;
        X[3][0] = 6.0; X[3][1] = 60.0;
        std::vector<double> y = {0.0, 0.0, 1.0, 1.0};

        DecisionStump stump(SplitCriterion::Gini, 1);
        stump.fit(X, y);
        assert(stump.has_split());
        assert(stump.split_feature() == 0); // Выбран меньший индекс признака (0 вместо 1)
        assert(approx_equal(stump.split_threshold(), 3.5));
    }

    std::cout << "[OK] Test 7: Граничные случаи раздела 9 (n=1, 1 класс, min_samples_leaf, детерминизм) пройдены\n";
}

} // namespace

int main() {
    try {
        std::cout << "=== Запуск модульных тестов Лабораторной работы №2 ===\n";
        test_compare_with_naive();
        test_separable_synthetic_data();
        test_constant_feature_ignored();
        test_duplicate_feature_values();
        test_polymorphic_imodel();
        test_all_required_exceptions();
        benchmark_on_real_dataset();
        test_edge_cases_section_9();
        std::cout << "=== Все тесты Лабораторной работы №2 успешно пройдены! ===\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Неперехваченное исключение в тестах: " << ex.what() << "\n";
        return 1;
    }
}