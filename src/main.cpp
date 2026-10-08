#include "dataset.hpp"
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
    } catch (const std::exception& ex) {
        std::cerr << "Ошибка: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
