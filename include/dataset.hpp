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

struct NumericFeatureInfo {
    std::size_t missing_count;
    double min;
    double max;
    double mean;
    double variance;
    double median;
    double q05;
    double q25;
    double q75;
    double q95;
    std::vector<size_t> histogram_counts;
};

struct CategoricalFeatureInfo {
    std::size_t missing_count;
    std::vector<std::string> categories;
    std::unordered_map<std::string, std::size_t> frequencies;
};


using FeatureInfo = std::variant<NumericFeatureInfo, CategoricalFeatureInfo>;
using DatasetInfo = std::unordered_map<std::string, FeatureInfo>;

class Dataset
{
private:
    std::unordered_map<std::string, Column> columns_;
    std::vector<std::string> feature_names_;

    std::size_t row_count_ =0;

public:
    static Dataset load(const std::filesystem::path& path, const CsvConfig& config);
    Column& get_column_mut(const std::string& name) { return columns_.at(name); }
    DatasetInfo analyze() const;
    std::size_t rows() const {return row_count_;}
    std::size_t cols() const {return feature_names_.size();}
    const std::vector<std::string>& feature_names() const {return feature_names_;}
    const Column& get_column(const std::string& name) const {return columns_.at(name);}
};
