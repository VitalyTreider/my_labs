#include "view.hpp"

void print_dataset_summary(const Dataset& ds, const DatasetInfo& info) {
    std::cout << "\n=========================================\n";
    std::cout << "           DATASET SUMMARY               \n";
    std::cout << "=========================================\n";
    std::cout << "Total rows: " << ds.rows() << "\n"; 
    std::cout << "Total cols: " << ds.cols() << "\n";
    std::cout << "-----------------------------------------\n";
    
    std::cout << std::left 
              << std::setw(15) << "Column Name" 
              << std::setw(15) << "Type" 
              << std::setw(15) << "Missing Values" << "\n";
    std::cout << "-----------------------------------------\n";

    int total_missing = 0;

    for (const auto& name : ds.feature_names()) {
        // Достаем УЖЕ ПОСЧИТАННУЮ статистику
        const FeatureInfo& f_info = info.at(name);
        
        std::string type_str;
        size_t missing_count = 0;

        // Проверяем, числовая она или текстовая
        if (std::holds_alternative<NumericFeatureInfo>(f_info)) {
            type_str = "Numeric";
            missing_count = std::get<NumericFeatureInfo>(f_info).missing_count;
        } else if (std::holds_alternative<CategoricalFeatureInfo>(f_info)) {
            type_str = "Categorical";
            missing_count = std::get<CategoricalFeatureInfo>(f_info).missing_count;
        }

        total_missing += missing_count;

        std::cout << std::left 
                  << std::setw(15) << name 
                  << std::setw(15) << type_str 
                  << std::setw(15) << missing_count << "\n";
    }

    std::cout << "=========================================\n";
    std::cout << "Total missing values in dataset: " << total_missing << "\n";
    std::cout << "=========================================\n\n";
}

void print_categorical_histogram(const std::string& col_name, const CategoricalFeatureInfo& stats) {
    std::cout << "Histogram for Categorical Column: " << col_name << "\n";
    std::cout << "--------------------------------------------------\n";
    
    size_t max_freq = 0;
    for (const auto& [cat, freq] : stats.frequencies) {
        if (freq > max_freq) max_freq = freq;
    }

    if (max_freq == 0) {
        std::cout << "No data to display.\n\n";
        return;
    }

    const int MAX_BAR_LENGTH = 50; 

    for (const auto& cat : stats.categories) {
        size_t freq = stats.frequencies.at(cat);
        int bar_length = static_cast<int>((static_cast<double>(freq) / max_freq) * MAX_BAR_LENGTH);
        std::string display_cat = cat.empty() ? "?" : cat;

        std::cout << std::left << std::setw(5) << display_cat << " | "
                  << std::string(bar_length, '*') 
                  << " (" << freq << ")\n";
    }
    std::cout << "--------------------------------------------------\n\n";
}

void print_numeric_histogram(const std::string& col_name, const NumericFeatureInfo& stats) {
    std::cout << "Histogram for Numeric Column: " << col_name << "\n";
    std::cout << "--------------------------------------------------\n";

    if (stats.histogram_counts.empty()) {
        std::cout << "No data to display.\n\n";
        return;
    }

    size_t max_freq = 0;
    for (size_t count : stats.histogram_counts) {
        if (count > max_freq) max_freq = count;
    }

    const int MAX_BAR_LENGTH = 50;
    const int NUM_BINS = 10;
    double bin_width = (stats.max - stats.min) / NUM_BINS;

    for (int i = 0; i < NUM_BINS; ++i) {
        double bin_start = stats.min + i * bin_width;
        double bin_end = bin_start + bin_width;
        size_t freq = stats.histogram_counts[i];

        int bar_length = max_freq == 0 ? 0 : static_cast<int>((static_cast<double>(freq) / max_freq) * MAX_BAR_LENGTH);

        std::cout << "[" << std::fixed << std::setprecision(2) << std::setw(6) << bin_start << ", "
                  << std::setw(6) << bin_end << ") | "
                  << std::string(bar_length, '*') 
                  << " (" << freq << ")\n";
    }
    std::cout << "--------------------------------------------------\n\n";
}