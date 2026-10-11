#pragma once
#include <iostream>
#include <iomanip>
#include "dataset.hpp"

void print_numeric_histogram(const std::string& col_name, const NumericFeatureInfo& stats);
void print_dataset_summary(const Dataset& ds, const DatasetInfo& info);
void print_categorical_histogram(const std::string& col_name, const CategoricalFeatureInfo& stats);