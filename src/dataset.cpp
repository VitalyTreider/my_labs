#include "dataset.hpp"
#include <charconv>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace {
    
std::string trim(const std::string& str)
{
    const auto first = str.find_first_not_of(" \t\t\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = str.find_last_not_of(" \t\t\n");
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

std::optional<double> try_parser_double(const std::string& str)
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
        throw std::runtime_error("Faild to open dataset file: " + path.string());
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

        for (std::size_t col_idx = 0; col_idx < dataset.feature_names_.size(); ++col_idx) {
            const auto& raw_col = raw_columns[col_idx];
            bool is_numeric = true;
            bool has_valid_values = false;

            for (const std::string& cell : raw_col) {
                if (cell.empty() || cell == config.missing_value) {
                    continue;
                }
                has_valid_values = true;
                if (!try_parser_double(cell).has_value()) {
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
                        num_col.values.push_back(try_parser_double(cell));
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
}