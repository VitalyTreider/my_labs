#include "dataset.hpp"

#include <charconv>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <algorithm>
#include <numeric>

namespace {


std::string trim(const std::string& str)
{
    const auto first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}


std::vector<std::string> split_line(const std::string& line, char separator)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream stream(line);

    while (std::getline(stream, token, separator)) {
        tokens.push_back(trim(token));
    }

    
    if (!line.empty() && line.back() == separator) {
        tokens.emplace_back("");
    }

    return tokens;
}


std::optional<double> try_parse_double(const std::string& str)
{
    if (str.empty()) {
        return std::nullopt;
    }

    double value = 0.0;
    const char* begin = str.data();
    const char* end = str.data() + str.size();

    
    auto [ptr, ec] = std::from_chars(begin, end, value);

    
    if (ec == std::errc{} && ptr == end) {
        return value;
    }
    return std::nullopt;
}

}

Dataset Dataset::load(const std::filesystem::path& path, const CsvConfig& config)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open dataset file: " + path.string());
    }

    Dataset dataset;
    std::string line;

    
    std::vector<std::vector<std::string>> raw_columns;
    bool is_first_line = true;

    while (std::getline(file, line)) {
        if (trim(line).empty()) {
            continue;
        }

        std::vector<std::string> cells = split_line(line, config.separator);

        if (is_first_line) {
            is_first_line = false;
            raw_columns.resize(cells.size());

            if (config.has_header) {
                dataset.feature_names_ = cells;
                continue;
            } else {
                for (std::size_t i = 0; i < cells.size(); ++i) {
                    dataset.feature_names_.push_back("Col_" + std::to_string(i));
                }
            }
        }

        if (cells.size() != dataset.feature_names_.size()) {
            throw std::runtime_error(
                "Column count mismatch at data row " + std::to_string(dataset.row_count_ + 1)
            );
        }

        for (std::size_t col_idx = 0; col_idx < cells.size(); ++col_idx) {
            raw_columns[col_idx].push_back(cells[col_idx]);
        }

        ++dataset.row_count_;
    }

    
    for (std::size_t col_idx = 0; col_idx < dataset.feature_names_.size(); ++col_idx) {
        const auto& raw_col = raw_columns[col_idx];
        bool is_numeric = true;
        bool has_valid_values = false;

        for (const std::string& cell : raw_col) {
            if (cell.empty() || cell == config.missing_value) {
                continue;
            }
            has_valid_values = true;
            if (!try_parse_double(cell).has_value()) {
                is_numeric = false;
                break;
            }
        }

        
        if (!has_valid_values) {
            is_numeric = false;
        }

        const std::string& col_name = dataset.feature_names_[col_idx];

        if (is_numeric) {
            NumericColumn num_col;
            num_col.values.reserve(raw_col.size());

            for (const std::string& cell : raw_col) {
                if (cell.empty() || cell == config.missing_value) {
                    num_col.values.push_back(std::nullopt);
                } else {
                    num_col.values.push_back(try_parse_double(cell));
                }
            }
            dataset.columns_.emplace(col_name, std::move(num_col));
        } else {
            CategoricalColumn cat_col;
            cat_col.values.reserve(raw_col.size());

            for (const std::string& cell : raw_col) {
                if (cell.empty() || cell == config.missing_value) {
                    cat_col.values.push_back(std::nullopt);
                } else {
                    cat_col.values.push_back(cell);
                }
            }
            dataset.columns_.emplace(col_name, std::move(cat_col));
        }
    }

    return dataset;
}

NumericStatistics compute_numeric_stats(const NumericColumn& col) {
    NumericStatistics stats;
    std::vector<double> valid_values;
    for (const auto& opt_val : col.values){
        if(opt_val.has_value()) {
            valid_values.push_back(opt_val.value());
        } else {
            ++stats.num_passes;
        }
    }

    if (valid_values.empty()){
        throw std::runtime_error("There is no data for calculation.");
    }

    std::sort(valid_values.begin(), valid_values.end());
    stats.min_val = valid_values.front();
    stats.max_val = valid_values.back();
    double sum = std::accumulate(valid_values.begin(), valid_values.end(), 0.0);
    stats.mean = sum / valid_values.size();
    double vars = 0;

    for (const auto opt_val : valid_values){
        vars += (opt_val - stats.mean) * (opt_val - stats.mean);
    }
    if (valid_values.size() > 1) {
        stats.variance = vars / (valid_values.size() - 1);
    } else {
        stats.variance = 0;
    }

    size_t mid = valid_values.size() / 2;
    if (valid_values.size() % 2 == 0) {
        stats.median = (valid_values[mid] + valid_values[mid-1]) / 2.0;
    } else {
        stats.median = valid_values[mid];
    }
    
    size_t in_q05 = static_cast<size_t>(valid_values.size() * 0.05);
    size_t in_q25 = static_cast<size_t>(valid_values.size() * 0.25);
    size_t in_q75 = static_cast<size_t>(valid_values.size() * 0.75);
    size_t in_q95 = static_cast<size_t>(valid_values.size() * 0.95);
    stats.q05 = valid_values[in_q05];
    stats.q25 = valid_values[in_q25];
    stats.q75 = valid_values[in_q75];
    stats.q95 = valid_values[in_q95];

    return stats;
}

CategoricalStatistics compute_categorical_stats(const CategoricalColumn& col) {
    CategoricalStatistics stats;
    for (const auto& opt_val : col.values){
        if(opt_val.has_value()) {
            stats.frequencies[opt_val.value()]++;
        } else {
            ++stats.num_passes;
        }
    }

    for (const auto& [category, count] : stats.frequencies) {
        stats.categories.push_back(category);
    }

    return stats;
}

void impute_missing_values(Dataset& ds) {
    for (const auto& name : ds.feature_names()) {
        Column& col = ds.get_column_mut(name);
        if (std::holds_alternative<NumericColumn>(col)){
            NumericColumn& num_col = std::get<NumericColumn>(col);
            NumericStatistics stats = compute_numeric_stats(num_col);

            for (auto& val : num_col.values) {
                if(!val.has_value()){
                    val = stats.median;
                }
            }
        } else if (std::holds_alternative<CategoricalColumn>(col)) {
            CategoricalColumn& cat_col = std::get<CategoricalColumn>(col);
            CategoricalStatistics stats = compute_categorical_stats(cat_col);

            std::string mode = "";
            size_t max_freq = 0;

            for (const auto& [category, freq] : stats.frequencies) {
                if (!category.empty() && freq > max_freq) {
                    mode = category;
                    max_freq = freq;
                }
            }

            for (auto& val : cat_col.values) {
                if (!val.has_value()) {
                    val = mode;
                }
            }
        }
    }
}