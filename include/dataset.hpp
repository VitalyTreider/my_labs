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

// стратегии предобработки
enum class NumericImputation { Mean, Median, Zero };
enum class Scaling { MinMax, Standard, Robust };
enum class Layout { RowMajor, ColumnMajor };
enum class CategoricalEncoding { OneHot, Ordinal };

// класс плотной числовой матрицы
class Matrix {
private:
    std::vector<double> data_;
    std::size_t rows_;
    std::size_t cols_;
    Layout layout_;

public:
    // конструктор
    Matrix(std::size_t rows, std::size_t cols, Layout layout = Layout::RowMajor)
        : rows_(rows), cols_(cols), layout_(layout) {
        data_.resize(rows * cols, 0.0);
    }

    // доступ к элементам матрицы
    double& operator()(std::size_t row, std::size_t column) {
        if (layout_ == Layout::RowMajor) {
            return data_[row * cols_ + column];
        } else {
            return data_[column * rows_ + row];
        }
    }

    const double& operator()(std::size_t row, std::size_t column) const {
        if (layout_ == Layout::RowMajor) {
            return data_[row * cols_ + column];
        } else {
            return data_[column * rows_ + row];
        }
    }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    const std::vector<double>& data() const { return data_; }
    Layout layout() const { return layout_; }

    // методы сериализации 
    void save(const std::filesystem::path& path) const;
    static Matrix load(const std::filesystem::path& path);
};

// финальный результат предобработки
struct TransformedDataset {
    Matrix matrix;
    std::vector<std::string> feature_names;
};

class Dataset
{
private:
    std::unordered_map<std::string, Column> columns_;
    std::vector<std::string> feature_names_;

    std::size_t row_count_ =0;

public:
    static Dataset load(const std::filesystem::path& path, const CsvConfig& config);
    Column& get_column_mut(const std::string& name) { return columns_.at(name); }

    DatasetInfo analyze(std::size_t num_bins = 10) const;

    void impute(const DatasetInfo& info, NumericImputation strategy);

    TransformedDataset transform(const DatasetInfo& info, Scaling scaling, Layout layout = Layout::RowMajor, CategoricalEncoding encoding = CategoricalEncoding::OneHot) const;

    std::size_t rows() const {return row_count_;}
    std::size_t cols() const {return feature_names_.size();}
    const std::vector<std::string>& feature_names() const {return feature_names_;}
    const Column& get_column(const std::string& name) const {return columns_.at(name);}
};