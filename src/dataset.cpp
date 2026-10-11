#include "dataset.hpp"

#include <span>
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



NumericFeatureInfo calculate_numeric_info(std::span<const double> valid_values, size_t missing_count) {
    NumericFeatureInfo stats{};
    stats.missing_count = missing_count;

    if (valid_values.empty()) {
        return stats;
    }

    // Копируем данные для сортировки (span менять нельзя)
    std::vector<double> sorted_vals(valid_values.begin(), valid_values.end());
    std::ranges::sort(sorted_vals);

    stats.min = sorted_vals.front();
    stats.max = sorted_vals.back();

    double sum = std::accumulate(sorted_vals.begin(), sorted_vals.end(), 0.0);
    stats.mean = sum / sorted_vals.size();

    double vars = 0;
    for (double val : sorted_vals) {
        vars += (val - stats.mean) * (val - stats.mean);
    }
    stats.variance = sorted_vals.size() > 1 ? vars / (sorted_vals.size() - 1) : 0.0;

    size_t mid = sorted_vals.size() / 2;
    stats.median = (sorted_vals.size() % 2 == 0) ?
                   (sorted_vals[mid] + sorted_vals[mid-1]) / 2.0 :
                   sorted_vals[mid];

    stats.q05 = sorted_vals[static_cast<size_t>(sorted_vals.size() * 0.05)];
    stats.q25 = sorted_vals[static_cast<size_t>(sorted_vals.size() * 0.25)];
    stats.q75 = sorted_vals[static_cast<size_t>(sorted_vals.size() * 0.75)];
    stats.q95 = sorted_vals[static_cast<size_t>(sorted_vals.size() * 0.95)];

    stats.histogram_counts.assign(10, 0);
    if (stats.max > stats.min) {
        double bin_width = (stats.max - stats.min) / 10.0;
        for (double val : valid_values) {
            int bin = static_cast<int>((val - stats.min) / bin_width);
            if (bin >= 10) bin = 9;
            stats.histogram_counts[bin]++;
        }
    } else {
        stats.histogram_counts[0] = valid_values.size();
    }

    return stats;
}


DatasetInfo Dataset::analyze() const {
    DatasetInfo info;

    for (const auto& name : feature_names_) {
        const Column& col = columns_.at(name);

        if (std::holds_alternative<NumericColumn>(col)) {
            const auto& num_col = std::get<NumericColumn>(col);
            std::vector<double> valid_values;
            size_t missing_count = 0;

            for (const auto& opt_val : num_col.values) {
                if (opt_val.has_value()) {
                    valid_values.push_back(opt_val.value());
                } else {
                    missing_count++;
                }
            }

            
            info[name] = calculate_numeric_info(std::span<const double>(valid_values), missing_count);

        } else if (std::holds_alternative<CategoricalColumn>(col)) {
            const auto& cat_col = std::get<CategoricalColumn>(col);
            CategoricalFeatureInfo cat_stats;
            cat_stats.missing_count = 0;

            for (const auto& opt_val : cat_col.values) {
                if (opt_val.has_value()) {
                    cat_stats.frequencies[opt_val.value()]++;
                } else {
                    cat_stats.missing_count++;
                }
            }

            for (const auto& [category, count] : cat_stats.frequencies) {
                cat_stats.categories.push_back(category);
            }

            info[name] = cat_stats;
        }
    }

    return info;
}