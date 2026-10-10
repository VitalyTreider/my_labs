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

int main() {
    try {
        test_dataset_loading();
        test_column_types();
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
        return 1; 
    }
    return 0;
}