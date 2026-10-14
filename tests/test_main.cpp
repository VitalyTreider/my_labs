#include "dataset.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <filesystem>
#include <fstream>

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
    Dataset ds = Dataset::load("data/credit_approval.csv", config);

    ASSERT_EQUAL(690, ds.rows());
    ASSERT_EQUAL(16, ds.cols());
    
    std::cout << "test_dataset_loading OK!\n";
}

void test_column_types() {
    CsvConfig config;
    Dataset ds = Dataset::load("data/credit_approval.csv", config);

    const Column& col_a2 = ds.get_column("A2");
    ASSERT_TRUE(std::holds_alternative<NumericColumn>(col_a2), "A2 should be NumericColumn");

    const Column& col_a1 = ds.get_column("A1");
    ASSERT_TRUE(std::holds_alternative<CategoricalColumn>(col_a1), "A1 should be CategoricalColumn");

    std::cout << "test_column_types OK!\n";
}

void test_numeric_stats() {
    CsvConfig config;
    Dataset ds = Dataset::load("data/credit_approval.csv", config);

    DatasetInfo info = ds.analyze();
    
    const FeatureInfo& f_info = info.at("A2");
    ASSERT_TRUE(std::holds_alternative<NumericFeatureInfo>(f_info), "A2 should be NumericFeatureInfo");

    const auto& stats = std::get<NumericFeatureInfo>(f_info);

    ASSERT_TRUE(stats.min < stats.max, "Min should be less than Max");
    ASSERT_TRUE(stats.mean > stats.min && stats.mean < stats.max, "Mean should be between Min and Max");
    ASSERT_TRUE(stats.median >= stats.min && stats.median <= stats.max, "Median should be valid");
    ASSERT_TRUE(stats.q05 <= stats.q25 && stats.q25 <= stats.q75 && stats.q75 <= stats.q95, "Quantiles should be in ascending order");

    std::cout << "test_numeric_stats OK!\n";
}

void test_categoric_stats() {
    CsvConfig config;
    Dataset ds = Dataset::load("data/credit_approval.csv", config);

    DatasetInfo info = ds.analyze();

    const FeatureInfo& f_info = info.at("A1");
    ASSERT_TRUE(std::holds_alternative<CategoricalFeatureInfo>(f_info), "A1 should be CategoricalFeatureInfo");

    const auto& stats = std::get<CategoricalFeatureInfo>(f_info);
    
    ASSERT_TRUE(stats.categories.size() > 0 && stats.frequencies.size() > 0, "Categorical stats should have categories and frequencies");
    std::cout << "test_categoric_stats OK!\n";
}

void test_binary_serialization() {
    // тестовая матрица 2х2
    Matrix original(2, 2, Layout::RowMajor);
    original(0, 0) = 1.5;
    original(0, 1) = 2.5;
    original(1, 0) = 3.5;
    original(1, 1) = 4.5;

    std::filesystem::path temp_path = "test_matrix.bin";

    original.save(temp_path);

    Matrix loaded = Matrix::load(temp_path);

    ASSERT_EQUAL(original.rows(), loaded.rows());
    ASSERT_EQUAL(original.cols(), loaded.cols());

    ASSERT_TRUE(loaded(0, 0) == 1.5, "Cell 0,0 mismatch");
    ASSERT_TRUE(loaded(0, 1) == 2.5, "Cell 0,1 mismatch");
    ASSERT_TRUE(loaded(1, 0) == 3.5, "Cell 1,0 mismatch");
    ASSERT_TRUE(loaded(1, 1) == 4.5, "Cell 1,1 mismatch");

    // удаляем временный файл
    std::filesystem::remove(temp_path);

    std::cout << "test_binary_serialization OK!\n";
}

void test_inconsistent_csv() {
    // создаем кривой файл (во второй строке только 1 колонка вместо 2)
    std::filesystem::path bad_csv = "bad_test.csv";
    std::ofstream out(bad_csv);
    out << "col1,col2\n";
    out << "val1,val2\n";
    out << "val1\n";
    out.close();

    bool caught_exception = false;
    try {
        CsvConfig config;
        Dataset ds = Dataset::load(bad_csv, config);
    } catch (const std::runtime_error& e) {
        caught_exception = true;
    }

    std::filesystem::remove(bad_csv);
    
    ASSERT_TRUE(caught_exception, "Should throw exception on inconsistent CSV");
    std::cout << "test_inconsistent_csv OK!\n";
}

int main() {
    try {
        test_dataset_loading();
        test_column_types();
        test_numeric_stats();
        test_categoric_stats();
        test_binary_serialization();
        test_inconsistent_csv();
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
        return 1; 
    }
    return 0;
}