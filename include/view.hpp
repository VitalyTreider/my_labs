#pragma once
#include <iostream>
#include <iomanip>
#include "dataset.hpp"


void print_dataset_summary(const Dataset& ds);
void print_categorical_histogram(const std::string& col_name, const CategoricalStatistics& stats);