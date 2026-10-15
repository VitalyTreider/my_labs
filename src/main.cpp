#include "dataset.hpp"
#include <cxxopts.hpp>
#include "view.hpp"
#include <iostream>
#include <string>

NumericImputation parse_imputation(const std::string& str) {
    if (str == "median") {
        return NumericImputation::Median;
    } else if (str == "mean") {
        return NumericImputation::Mean;
    } else if (str == "zero") {
        return NumericImputation::Zero;
    } else {
        throw std::invalid_argument("Неизвестная стратегия: " + str);
    }
}

Scaling parse_scaling(const std::string& str) {
    if (str == "minmax") {
        return Scaling::MinMax;
    } else if (str == "robust") {
        return Scaling::Robust;
    } else if (str == "standard") {
        return Scaling::Standard;
    } else {
        throw std::invalid_argument("Неизвестная стратегия: " + str);
    }
}

CategoricalEncoding parse_encoding(const std::string& str) {
    if (str == "onehot") {
        return CategoricalEncoding::OneHot;
    } else if (str == "ordinal") {
        return CategoricalEncoding::Ordinal;
    } else {
        throw std::invalid_argument("Неизвестная стратегия: " + str);
    }
}

int main(int argc, char* argv[]) {
    try {
        cxxopts::Options options("lab1_app", "Лабораторная работа 1: анализ и предобработка данных");

        options.add_options()
            ("i,input", "Путь к входному CSV-файлу", cxxopts::value<std::string>()->default_value("data/credit_approval.csv"))
            ("b,bin", "Путь для сохранения бинарного файла", cxxopts::value<std::string>()->default_value("data/dataset.bin"))
            ("n,histogram", "Число интервалов гистограммы", cxxopts::value<std::size_t>()->default_value("10"))
            ("m,imputation", "Стратегия заполнения числовых пропусков", cxxopts::value<std::string>()->default_value("median"))
            ("s,scaling", "Способ масштабирования", cxxopts::value<std::string>()->default_value("minmax"))
            ("e,encoding", "Спсоб кодирования категорий", cxxopts::value<std::string>()->default_value("onehot"))
            ("h,help", "Вывод справки по параметрам");

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help() << "\n";
            return 0;
        }

        std::string input_path = result["input"].as<std::string>();
        std::string bin = result["bin"].as<std::string>();
        std::size_t num_bins = result["histogram"].as<std::size_t>();
        if (num_bins == 0){
            throw std::invalid_argument("Число интервалов гистограммы не может быть равно 0!");
        }
        NumericImputation imt = parse_imputation(result["imputation"].as<std::string>());
        Scaling sc = parse_scaling(result["scaling"].as<std::string>());
        CategoricalEncoding en = parse_encoding(result["encoding"].as<std::string>());
    


        CsvConfig config;
        Dataset ds = Dataset::load(input_path, config);

        std::cout << "Загружено строк: " << ds.rows() << "\n";
        std::cout << "Загружено столбцов: " << ds.cols() << "\n\n";

        
        DatasetInfo info = ds.analyze(num_bins);

        
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
            ds.impute(info, imt);
            
            // масштабируем числа и кодируем текст
            TransformedDataset td = ds.transform(info, sc, Layout::RowMajor, en);
            
            // сохраняем в бин файл
            std::filesystem::path bin_path = bin;
            td.matrix.save(bin_path);
            
            std::cout << "Предобработка завершена!\n";
            std::cout << "Размеры матрицы: " << td.matrix.rows() << " x " << td.matrix.cols() << "\n";
            
            std::cout << "CSV:     " << std::filesystem::file_size(input_path) << " байт\n";
            std::cout << "Binary:  " << std::filesystem::file_size(bin_path) << " байт\n";
        }

    } catch (const std::exception& ex) {
        std::cerr << "Ошибка: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}