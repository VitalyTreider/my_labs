#include "dataset.hpp"
#include "decision_stump.hpp"
#include "exceptions.hpp"
#include "model.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

// доля правильных ответов (классификация)
double compute_accuracy(std::span<const double> y_true, std::span<const double> y_pred) {
    if (y_true.size() != y_pred.size() || y_true.empty()) {
        return 0.0;
    }
    std::size_t correct = 0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == y_pred[i]) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(y_true.size());
}

//среднеквадратичная ошибка MSE (регрессия)
double compute_mse(std::span<const double> y_true, std::span<const double> y_pred) {
    if (y_true.size() != y_pred.size() || y_true.empty()) {
        return 0.0;
    }
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        const double diff = y_true[i] - y_pred[i];
        sum_sq += diff * diff;
    }
    return sum_sq / static_cast<double>(y_true.size());
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        std::string input_path = "data/credit_approval.csv";
        std::string target_col = "A16";
        double train_ratio = 0.8;
        std::size_t min_samples_leaf = 5;
        SplitCriterion criterion = SplitCriterion::Gini;

        // фргументы командной строки
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
                input_path = argv[++i];
            } else if ((arg == "-t" || arg == "--target") && i + 1 < argc) {
                target_col = argv[++i];
            } else if ((arg == "-r" || arg == "--ratio") && i + 1 < argc) {
                train_ratio = std::stod(argv[++i]);
            } else if ((arg == "-l" || arg == "--leaf") && i + 1 < argc) {
                min_samples_leaf = static_cast<std::size_t>(std::stoul(argv[++i]));
            } else if ((arg == "-c" || arg == "--criterion") && i + 1 < argc) {
                std::string c = argv[++i];
                if (c == "gini") {
                    criterion = SplitCriterion::Gini;
                } else if (c == "variance") {
                    criterion = SplitCriterion::Variance;
                } else {
                    throw InvalidModelArgumentError("Unknown criterion: " + c + " (use 'gini' or 'variance')");
                }
            } else if (arg == "-h" || arg == "--help") {
                std::cout << "Использование: " << argv[0] << " [опции]\n"
                          << "  -i, --input <path>       Путь к CSV (по умолчанию: data/credit_approval.csv)\n"
                          << "  -t, --target <name>      Имя целевого столбца (по умолчанию: A16)\n"
                          << "  -r, --ratio <0..1>       Доля обучающей выборки (по умолчанию: 0.8)\n"
                          << "  -l, --leaf <int>         Мин. размер листа min_samples_leaf (по умолчанию: 5)\n"
                          << "  -c, --criterion <type>   Критерий: gini или variance (по умолчанию: gini)\n";
                return 0;
            }
        }

        //загрузка датасета, деление на train/test
        CsvConfig cfg;
        Dataset full_ds = Dataset::load(input_path, cfg);
        auto [train_ds, test_ds] = full_ds.split(train_ratio);

        // стата по обуч выборке
        DatasetInfo train_info = train_ds.analyze();

        //заполняем выборки
        train_ds.impute(train_info, NumericImputation::Median);
        test_ds.impute(train_info, NumericImputation::Median);

        TransformedDataset train_td = train_ds.transform(
            train_info, Scaling::MinMax, Layout::RowMajor, CategoricalEncoding::OneHot, target_col
        );
        TransformedDataset test_td = test_ds.transform(
            train_info, Scaling::MinMax, Layout::RowMajor, CategoricalEncoding::OneHot, target_col
        );

        //обучаем пень
        DecisionStump stump(criterion, min_samples_leaf);
        IModel& model = stump;

        const auto start_time = std::chrono::high_resolution_clock::now();
        model.fit(train_td.matrix, train_td.target);
        const auto end_time = std::chrono::high_resolution_clock::now();
        const double fit_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        //делаем предсказания
        std::vector<double> train_preds = model.predict(train_td.matrix);
        std::vector<double> test_preds  = model.predict(test_td.matrix);

        //резы
        std::cout << "=== Обучение DecisionStump завершено ===\n";
        std::cout << "Обучающая выборка: " << train_td.matrix.rows() << " строк x "
                  << train_td.matrix.cols() << " признаков\n";
        std::cout << "Тестовая выборка:  " << test_td.matrix.rows() << " строк x "
                  << test_td.matrix.cols() << " признаков\n";
        std::cout << "Критерий:          "
                  << (criterion == SplitCriterion::Gini ? "Gini (классификация)" : "Variance (регрессия)") << "\n";
        std::cout << "Время обучения:    " << fit_ms << " мс\n\n";
        std::cout << "min_samples_leaf:  " << min_samples_leaf << "\n\n";

        const std::size_t preview_count = std::min<std::size_t>(10, test_preds.size());
        std::cout << "Первые " << preview_count << " объектов тестовой выборки:\n";
        std::cout << "------------------------------------------\n";
        std::cout << std::left
                  << std::setw(10) << "Объект "
                  << std::setw(16) << "Истинный y "
                  << std::setw(16) << "Предсказание" << "\n";
        std::cout << "------------------------------------------\n";
        for (std::size_t i = 0; i < preview_count; ++i) {
            std::cout << std::left
                      << std::setw(10) << i
                      << std::setw(16) << test_td.target[i]
                      << std::setw(16) << test_preds[i] << "\n";
        }
        std::cout << "------------------------------------------\n\n";

        if (stump.has_split()) {
            const std::size_t feat_idx = stump.split_feature();
            const std::string& feat_name = train_td.feature_names[feat_idx];
            std::cout << "Найденное правило разбиения:\n";
            std::cout << "  Признак:         [" << feat_idx << "] " << feat_name << "\n";
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "  Порог (t):       " << stump.split_threshold() << "\n";
            std::cout << "  Левый лист (<=): " << stump.left_value() << "\n";
            std::cout << "  Правый лист (>): " << stump.right_value() << "\n\n";
        } else {
            std::cout << "Допустимое разбиение не найдено (константный лист): "
                      << stump.left_value() << "\n\n";
        }

        std::cout << std::fixed << std::setprecision(4);
        if (criterion == SplitCriterion::Gini) {
            std::cout << "Train Accuracy:    " << compute_accuracy(train_td.target, train_preds) * 100.0 << "%\n";
            std::cout << "Test Accuracy:     " << compute_accuracy(test_td.target, test_preds) * 100.0 << "%\n";
        } else {
            std::cout << "Train MSE:         " << compute_mse(train_td.target, train_preds) << "\n";
            std::cout << "Test MSE:          " << compute_mse(test_td.target, test_preds) << "\n";
        }

    } catch (const std::exception& ex) {
        std::cerr << "Ошибка: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}