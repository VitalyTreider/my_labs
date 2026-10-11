#include "dataset.hpp"
#include "view.hpp"
#include <iostream>

int main() {
    try {
        CsvConfig config;
        Dataset ds = Dataset::load("data/credit_approval.csv", config);

        std::cout << "Загружено строк: " << ds.rows() << "\n";
        std::cout << "Загружено столбцов: " << ds.cols() << "\n\n";

        for (const auto& name : ds.feature_names()) {
            const Column& col = ds.get_column(name);

            if (std::holds_alternative<NumericColumn>(col)) {
                std::cout << "Столбец " << name << ": Числовой (Numeric)\n";
            } else {
                std::cout << "Столбуц " << name << ": Категориальный (Categorical)\n";
            }
        }
        print_dataset_summary(ds);
        const auto& col_a1 = std::get<CategoricalColumn>(ds.get_column("A1"));
        CategoricalStatistics stats_a1 = compute_categorical_stats(col_a1);
        print_categorical_histogram("A1", stats_a1);
    } catch (const std::exception& ex) {
        std::cerr << "Ошибка: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
