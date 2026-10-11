#pragma once

#include<string>
#include<vector>
#include<optional>
#include<variant>
#include<unordered_map>
#include<filesystem>

struct CsvConfig
{
    char separator = ',';
    std::string missing_value = "?";
    bool has_header = true;
};

struct NumericColumn
{
    std::vector<std::optional<double>> values;
};

struct CategoricalColumn
{
    std::vector<std::optional<std::string>> values;
};

using Column = std::variant<NumericColumn, CategoricalColumn>;

class Dataset
{
private:
    std::unordered_map<std::string, Column> columns_;
    std::vector<std::string> feature_names_;

    std::size_t row_count_ =0;

public:
    static Dataset load(const std::filesystem::path& path, const CsvConfig& config);

    std::size_t rows() const {return row_count_;}
    std::size_t cols() const {return feature_names_.size();}
    const std::vector<std::string>& feature_names() const {return feature_names_;}
    const Column& get_column(const std::string& name) const {return columns_.at(name);}
};

struct NumericStatistics {
    double num_passes;
    double min_val;
    double max_val;
    double mean;
    double median;
    double variance;
    double q05;
    double q25;
    double q75;
    double q95;
};

NumericStatistics compute_numeric_stats(const NumericColumn& col);