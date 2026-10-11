#include "view.hpp"

void print_dataset_summary(const Dataset& ds) {
    // ВЕРХУШКА ТАБЛИЦЫ
    std::cout << "\n=========================================\n";
    std::cout << "           DATASET SUMMARY               \n";
    std::cout << "=========================================\n";
    // Замени ds.rows_count() и ds.cols_count() на свои методы, если они называются иначе
    std::cout << "Total rows: " << ds.rows() << "\n"; 
    std::cout << "Total cols: " << ds.cols() << "\n";
    std::cout << "-----------------------------------------\n";
    
    // ЗАГОЛОВОК КОЛОНОК
    std::cout << std::left 
              << std::setw(15) << "Column Name" 
              << std::setw(12) << "Type" 
              << std::setw(10) << "Missing Values" << "\n";
    std::cout << "-----------------------------------------\n";

    int total_missing = 0;

    // ЦИКЛ ПО КОЛОНКАМ (адаптируй под свой метод итерации)
    // Предположим, у тебя есть геттер имен колонок ds.get_column_names()
    for (const auto& name : ds.feature_names()) {
        const auto& col = ds.get_column(name);
        
        std::string type_str;
        size_t missing_count = 0;

        // Проверяем тип колонки через std::holds_alternative
        if (std::holds_alternative<NumericColumn>(col)) {
            type_str = "Numeric";
            const auto& num_col = std::get<NumericColumn>(col);
            NumericStatistics stats = compute_numeric_stats(num_col);
            missing_count = stats.num_passes;
        } else if (std::holds_alternative<CategoricalColumn>(col)) {
            type_str = "Categorical";
            const auto& cat_col = std::get<CategoricalColumn>(col);
            CategoricalStatistics stats = compute_categorical_stats(cat_col);
            missing_count = stats.num_passes;
        }

        total_missing += missing_count;

        // ВЫВОД СТРОКИ
        std::cout << std::left 
                  << std::setw(15) << name 
                  << std::setw(15) << type_str 
                  << std::setw(15) << missing_count << "\n";
    }

    std::cout << "=========================================\n";
    std::cout << "Total missing values in dataset: " << total_missing << "\n";
    std::cout << "=========================================\n\n";
}

void print_categorical_histogram(const std::string& col_name, const CategoricalStatistics& stats) {
    std::cout << "Histogram for Categorical Column: " << col_name << "\n";
    std::cout << "--------------------------------------------------\n";
    
    // Ищем максимальную частоту для масштабирования графика
    size_t max_freq = 0;
    for (const auto& [cat, freq] : stats.frequencies) {
        if (freq > max_freq) max_freq = freq;
    }

    if (max_freq == 0) {
        std::cout << "No data to display.\n\n";
        return;
    }

    const int MAX_BAR_LENGTH = 50; // Максимальное количество звездочек в строке

    // Выводим категории по порядку (из вектора stats.categories)
    for (const auto& cat : stats.categories) {
        size_t freq = stats.frequencies.at(cat);
        
        // Масштабируем длину полоски
        int bar_length = static_cast<int>((static_cast<double>(freq) / max_freq) * MAX_BAR_LENGTH);
        
        // Если категория пустая (пропуск), выводим "?"
        std::string display_cat = cat.empty() ? "?" : cat;

        std::cout << std::left << std::setw(5) << display_cat << " | "
                  << std::string(bar_length, '*') 
                  << " (" << freq << ")\n";
    }
    std::cout << "--------------------------------------------------\n\n";
}