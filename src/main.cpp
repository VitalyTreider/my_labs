#include "dataset.hpp"
#include "view.hpp"
#include <iostream>
#include <string>

int main() {
    try {
        CsvConfig config;
        Dataset ds = Dataset::load("data/credit_approval.csv", config);

        std::cout << "Загружено строк: " << ds.rows() << "\n";
        std::cout << "Загружено столбцов: " << ds.cols() << "\n\n";

        
        DatasetInfo info = ds.analyze();

        
        print_dataset_summary(ds, info);

        
        std::string input;
        while (true) {
            std::cout << "Введите имя признака для просмотра (или 'q' для выхода): ";
            std::cin >> input;

            if (input == "q" || input == "quit" || input == "exit") {
                break;
            }

            
            if (info.find(input) == info.end()) {
                std::cout << "Ошибка: Признак '" << input << "' не найден. Попробуйте еще раз.\n\n";
                continue;
            }

            const FeatureInfo& f_info = info.at(input);
            
            
            if (std::holds_alternative<NumericFeatureInfo>(f_info)) {
                const auto& num_stats = std::get<NumericFeatureInfo>(f_info);
                std::cout << "\nПризнак: " << input << " (числовой)\n";
                std::cout << "  пропусков: " << num_stats.missing_count << "\n";
                std::cout << "  min: " << num_stats.min << "   max: " << num_stats.max << "\n";
                std::cout << "  mean: " << num_stats.mean << "  var: " << num_stats.variance << "\n\n";
                std::cout << "  q05: " << num_stats.q05 << "\n";
                std::cout << "  q25: " << num_stats.q25 << "\n";
                std::cout << "  q50 (median): " << num_stats.median << "\n";
                std::cout << "  q75: " << num_stats.q75 << "\n";
                std::cout << "  q95: " << num_stats.q95 << "\n\n";
                print_numeric_histogram(input, num_stats);
            
            } else if (std::holds_alternative<CategoricalFeatureInfo>(f_info)) {
                const auto& cat_stats = std::get<CategoricalFeatureInfo>(f_info);
                std::cout << "\nПризнак: " << input << " (категориальный)\n";
                std::cout << "  пропусков: " << cat_stats.missing_count << "\n\n";
                print_categorical_histogram(input, cat_stats);
            }
        }

        std::cout << "\n--- Завершение работы, цикл остановлен ---\n";

        std::string answer;
        std::cout << "\nСохранить преобразованный датасет? (y/n): ";
        std::cin >> answer;

        if (answer == "y" || answer == "Y") {
            std::cout << "Выполняем предобработку...\n";
            
            // заполняем пропуски
            ds.impute(info, NumericImputation::Median);
            
            // масштабируем числа и кодируем текст
            TransformedDataset td = ds.transform(info, Scaling::MinMax, Layout::RowMajor);
            
            // сохраняем в бин файл
            std::filesystem::path bin_path = "data/dataset.bin";
            td.matrix.save(bin_path);
            
            std::cout << "Предобработка завершена!\n";
            std::cout << "Размеры матрицы: " << td.matrix.rows() << " x " << td.matrix.cols() << "\n";
            
            std::cout << "CSV:     " << std::filesystem::file_size("data/credit_approval.csv") << " байт\n";
            std::cout << "Binary:  " << std::filesystem::file_size(bin_path) << " байт\n";
        }

    } catch (const std::exception& ex) {
        std::cerr << "Ошибка: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}