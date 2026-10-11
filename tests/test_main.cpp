#include "dataset.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

#define ASSERT_EQUAL(expected, actual) \
    if ((expected) != (actual)) { \
        throw std::runtime_error("Assertion failed: " + std::to_string(expected) + " != " + std::to_string(actual)); \
    }
#define ASSERT_TRUE(condition, msg) \
    if (!(condition)) { \
        throw std::runtime_error(std::string("Assertion failed: ") + msg); \
    }

void test_dataset_loading() {
    CsvConfig config;
    Dataset ds = Dataset::load("../data/credit_approval.csv", config);

    ASSERT_EQUAL(690, ds.rows());
    ASSERT_EQUAL(16, ds.cols());
    
    std::cout << "test_dataset_loading OK!\n";
}

void test_column_types() {
    CsvConfig config;
    Dataset ds = Dataset::load("../data/credit_approval.csv", config);

    const Column& col_a2 = ds.get_column("A2");
    ASSERT_TRUE(std::holds_alternative<NumericColumn>(col_a2), "A2 should be NumericColumn");

    const Column& col_a1 = ds.get_column("A1");
    ASSERT_TRUE(std::holds_alternative<CategoricalColumn>(col_a1), "A1 should be CategoricalColumn");

    std::cout << "test_column_types OK!\n";
}

void test_numeric_stats() {
    CsvConfig config;
    Dataset ds = Dataset::load("../data/credit_approval.csv", config);

    // Новый способ: получаем всю статистику один раз
    DatasetInfo info = ds.analyze();
    
    const FeatureInfo& f_info = info.at("A2");
    ASSERT_TRUE(std::holds_alternative<NumericFeatureInfo>(f_info), "A2 should be NumericFeatureInfo");

    const auto& stats = std::get<NumericFeatureInfo>(f_info);

    // В новых структурах переменные называются min и max
    ASSERT_TRUE(stats.min < stats.max, "Min should be less than Max");
    ASSERT_TRUE(stats.mean > stats.min && stats.mean < stats.max, "Mean should be between Min and Max");
    ASSERT_TRUE(stats.median >= stats.min && stats.median <= stats.max, "Median should be valid");
    ASSERT_TRUE(stats.q05 <= stats.q25 && stats.q25 <= stats.q75 && stats.q75 <= stats.q95, "Quantiles should be in ascending order");

    std::cout << "test_numeric_stats OK!\n";
}

void test_categoric_stats() {
    CsvConfig config;
    Dataset ds = Dataset::load("../data/credit_approval.csv", config);

    // Новый способ: получаем всю статистику один раз
    DatasetInfo info = ds.analyze();

    const FeatureInfo& f_info = info.at("A1");
    ASSERT_TRUE(std::holds_alternative<CategoricalFeatureInfo>(f_info), "A1 should be CategoricalFeatureInfo");

    const auto& stats = std::get<CategoricalFeatureInfo>(f_info);
    
    ASSERT_TRUE(stats.categories.size() > 0 && stats.frequencies.size() > 0, "Categorical stats should have categories and frequencies");
    std::cout << "test_categoric_stats OK!\n";
}

int main() {
    try {
        test_dataset_loading();
        test_column_types();
        test_numeric_stats();
        test_categoric_stats();
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
        return 1; 
    }
    return 0;
}