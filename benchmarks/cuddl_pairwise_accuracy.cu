#include <CLI/CLI.hpp>
#include <cuddl/cuddl.cuh>
#include <cuddl/fastx.hpp>

#include <thrust/device_vector.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t k_kmer_length = 25;
constexpr size_t k_bucket_count = 2048;

using sketch_type = cuddl::sketch<k_kmer_length, k_bucket_count>;

constexpr std::array<std::string_view, 13> k_case_header{
    "generator_seed",
    "power",
    "trial",
    "size_ratio",
    "requested_ani",
    "actual_ani",
    "mutation_count",
    "reference_bases",
    "query_bases",
    "reference_sha256",
    "query_sha256",
    "reference_path",
    "query_path",
};

struct case_record {
    uint64_t generator_seed;
    uint32_t power;
    uint32_t trial;
    uint32_t size_ratio;
    double requested_ani;
    double actual_ani;
    uint64_t mutation_count;
    uint64_t reference_bases;
    uint64_t query_bases;
    std::string reference_sha256;
    std::string query_sha256;
    std::string reference_path;
    std::string query_path;
};

struct metrics {
    double containment;
    double completeness;
    double wkid;
    double ani;
};

struct exact_metrics {
    metrics values;
    double set_derived_ani;
};

struct error_samples {
    std::vector<double> containment;
    std::vector<double> completeness;
    std::vector<double> wkid;
    std::vector<double> ani;
};

struct prepared_sequence {
    std::string path;
    std::string sha256;
    uint64_t bases;
    std::vector<uint64_t> unique_kmers;
    sketch_type sketch;
};

[[nodiscard]] std::vector<std::string> parse_csv_line(std::string const& input) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (size_t index = 0; index < input.size(); ++index) {
        auto const ch = input[index];
        if (ch == '"') {
            if (quoted && index + 1U < input.size() && input[index + 1U] == '"') {
                field.push_back('"');
                ++index;
            } else {
                quoted = !quoted;
            }
        } else if (ch == ',' && !quoted) {
            fields.push_back(std::move(field));
            field.clear();
        } else {
            field.push_back(ch);
        }
    }
    if (quoted) {
        throw std::runtime_error("unterminated quoted CSV field");
    }
    fields.push_back(std::move(field));
    return fields;
}

[[nodiscard]] uint64_t parse_u64(std::string const& value, char const* name, size_t row) {
    try {
        if (value.empty() || value.front() == '-') {
            throw std::invalid_argument("not unsigned");
        }
        size_t end = 0;
        auto const parsed = std::stoull(value, &end);
        if (end != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return parsed;
    } catch (std::exception const&) {
        throw std::runtime_error(
            "invalid " + std::string(name) + " at cases CSV row " + std::to_string(row)
        );
    }
}

[[nodiscard]] uint32_t parse_u32(std::string const& value, char const* name, size_t row) {
    auto const parsed = parse_u64(value, name, row);
    if (parsed > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(
            std::string(name) + " exceeds uint32 at cases CSV row " + std::to_string(row)
        );
    }
    return static_cast<uint32_t>(parsed);
}

[[nodiscard]] double parse_fraction(std::string const& value, char const* name, size_t row) {
    try {
        size_t end = 0;
        auto const parsed = std::stod(value, &end);
        if (end != value.size() || !std::isfinite(parsed) || parsed < 0.0 || parsed > 1.0) {
            throw std::invalid_argument("not a fraction");
        }
        return parsed;
    } catch (std::exception const&) {
        throw std::runtime_error(
            "invalid " + std::string(name) + " at cases CSV row " + std::to_string(row)
        );
    }
}

[[nodiscard]] case_record parse_case(std::vector<std::string> const& fields, size_t row) {
    if (fields.size() != k_case_header.size()) {
        throw std::runtime_error(
            "expected " + std::to_string(k_case_header.size()) + " columns at cases CSV row " +
            std::to_string(row) + ", got " + std::to_string(fields.size())
        );
    }
    case_record result{
        .generator_seed = parse_u64(fields[0], "generator_seed", row),
        .power = parse_u32(fields[1], "power", row),
        .trial = parse_u32(fields[2], "trial", row),
        .size_ratio = parse_u32(fields[3], "size_ratio", row),
        .requested_ani = parse_fraction(fields[4], "requested_ani", row),
        .actual_ani = parse_fraction(fields[5], "actual_ani", row),
        .mutation_count = parse_u64(fields[6], "mutation_count", row),
        .reference_bases = parse_u64(fields[7], "reference_bases", row),
        .query_bases = parse_u64(fields[8], "query_bases", row),
        .reference_sha256 = fields[9],
        .query_sha256 = fields[10],
        .reference_path = fields[11],
        .query_path = fields[12],
    };
    if (result.size_ratio == 0U) {
        throw std::runtime_error(
            "size_ratio must be positive at cases CSV row " + std::to_string(row)
        );
    }
    if (result.reference_bases == 0U || result.query_bases == 0U) {
        throw std::runtime_error(
            "sequence base counts must be positive at cases CSV row " + std::to_string(row)
        );
    }
    if (result.reference_path.empty() || result.query_path.empty()) {
        throw std::runtime_error(
            "FASTA paths must not be empty at cases CSV row " + std::to_string(row)
        );
    }
    return result;
}

void require_header(std::vector<std::string> const& fields) {
    if (fields.size() != k_case_header.size()) {
        throw std::runtime_error("cases CSV has an unexpected header");
    }
    for (size_t index = 0; index < fields.size(); ++index) {
        if (fields[index] != k_case_header[index]) {
            throw std::runtime_error(
                "cases CSV column " + std::to_string(index + 1U) + " must be " +
                std::string(k_case_header[index])
            );
        }
    }
}

[[nodiscard]] thrust::device_vector<uint64_t> copy_to_device(std::vector<uint64_t> const& host) {
    return {host.begin(), host.end()};
}

[[nodiscard]] prepared_sequence
prepare_sequence(std::string const& path, std::string const& sha256, uint64_t expected_bases) {
    auto parsed = CUDDL_UNWRAP(cuddl::parse_fasta_file(path, k_kmer_length));
    if (parsed.bases != expected_bases) {
        throw std::runtime_error(
            "FASTA base count differs from cases CSV for " + path + ": expected " +
            std::to_string(expected_bases) + ", parsed " + std::to_string(parsed.bases)
        );
    }
    if (parsed.kmers.empty()) {
        throw std::runtime_error("FASTA contains no valid 25-mers: " + path);
    }

    auto device = copy_to_device(parsed.kmers);
    sketch_type sketch;
    CUDDL_UNWRAP(sketch.add(device));

    std::sort(parsed.kmers.begin(), parsed.kmers.end());
    parsed.kmers.erase(std::unique(parsed.kmers.begin(), parsed.kmers.end()), parsed.kmers.end());
    return {
        .path = path,
        .sha256 = sha256,
        .bases = parsed.bases,
        .unique_kmers = std::move(parsed.kmers),
        .sketch = std::move(sketch),
    };
}

[[nodiscard]] size_t
intersection_size(std::vector<uint64_t> const& left, std::vector<uint64_t> const& right) {
    size_t intersection = 0;
    size_t left_index = 0;
    size_t right_index = 0;
    while (left_index < left.size() && right_index < right.size()) {
        if (left[left_index] < right[right_index]) {
            ++left_index;
        } else if (right[right_index] < left[left_index]) {
            ++right_index;
        } else {
            ++intersection;
            ++left_index;
            ++right_index;
        }
    }
    return intersection;
}

[[nodiscard]] exact_metrics
exact_pair_metrics(size_t left, size_t right, size_t intersection, double actual_ani) {
    auto const shared = static_cast<double>(intersection);
    auto const wkid = shared / static_cast<double>(std::min(left, right));
    return {
        .values =
            {
                .containment = shared / static_cast<double>(left),
                .completeness =
                    std::min(1.0, static_cast<double>(left) / static_cast<double>(right)),
                .wkid = wkid,
                .ani = actual_ani,
            },
        .set_derived_ani =
            wkid > 0.0 ? std::pow(wkid, 1.0 / static_cast<double>(k_kmer_length)) : 0.0,
    };
}

[[nodiscard]] double require(std::optional<double> value, char const* name) {
    if (!value) {
        throw std::runtime_error(std::string(name) + " was undefined for non-empty inputs");
    }
    return *value;
}

[[nodiscard]] metrics
sketch_metrics(sketch_type const& left, cuddl::pairwise_summary const& summary) {
    auto const ref = left.ref();
    return {
        .containment = require(ref.containment(summary), "containment"),
        .completeness = require(ref.completeness(summary), "completeness"),
        .wkid = require(ref.wkid(summary), "WKID"),
        .ani = require(ref.ani(summary), "ANI"),
    };
}

void emit_csv_field(std::ofstream& csv, std::string const& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) {
        csv << value;
        return;
    }
    csv << '"';
    for (auto const ch : value) {
        if (ch == '"') {
            csv << '"';
        }
        csv << ch;
    }
    csv << '"';
}

void emit_metric(
    std::ofstream& csv,
    double exact,
    double estimate,
    std::vector<double>& errors,
    bool record_error = true
) {
    auto const signed_error = estimate - exact;
    auto const absolute_error = std::abs(signed_error);
    csv << ',' << exact << ',' << estimate << ',' << signed_error << ',' << absolute_error;
    if (record_error) {
        errors.push_back(absolute_error);
    }
}

void emit_orientation(
    std::ofstream& csv,
    case_record const& input,
    sketch_type const& left,
    sketch_type const& right,
    char const* orientation,
    size_t left_size,
    size_t right_size,
    size_t intersection,
    error_samples& errors
) {
    auto const summary = CUDDL_UNWRAP(left.compare(right.ref()));
    auto const exact = exact_pair_metrics(left_size, right_size, intersection, input.actual_ani);
    auto const estimate = sketch_metrics(left, summary);
    auto const& counts = summary.counts;
    auto const primary_orientation = std::string_view{orientation} == "query_to_reference";

    if (counts.lower + counts.equal + counts.higher + counts.both_empty != k_bucket_count) {
        throw std::runtime_error("pairwise counts do not sum to the bucket count");
    }

    csv << "cuddl," << input.generator_seed << ',' << k_kmer_length << ',' << k_bucket_count << ','
        << input.power << ',' << input.trial << ',' << input.size_ratio << ','
        << input.requested_ani << ',' << input.actual_ani << ',' << input.mutation_count << ','
        << input.reference_bases << ',' << input.query_bases << ',';
    emit_csv_field(csv, input.reference_sha256);
    csv << ',';
    emit_csv_field(csv, input.query_sha256);
    csv << ',';
    emit_csv_field(csv, input.reference_path);
    csv << ',';
    emit_csv_field(csv, input.query_path);
    csv << ',' << orientation << ',' << left_size << ',' << right_size << ',' << intersection << ','
        << counts.lower << ',' << counts.equal << ',' << counts.higher << ',' << counts.both_empty;
    emit_metric(csv, exact.values.containment, estimate.containment, errors.containment);
    emit_metric(csv, exact.values.completeness, estimate.completeness, errors.completeness);
    emit_metric(csv, exact.values.wkid, estimate.wkid, errors.wkid, primary_orientation);
    csv << ',' << exact.set_derived_ani;
    emit_metric(csv, exact.values.ani, estimate.ani, errors.ani, primary_orientation);
    csv << '\n';
}

[[nodiscard]] double quantile(std::vector<double> const& sorted, double q) {
    auto const position = q * static_cast<double>(sorted.size() - 1U);
    auto const lower = static_cast<size_t>(position);
    auto const upper = std::min(lower + 1U, sorted.size() - 1U);
    auto const fraction = position - static_cast<double>(lower);
    return sorted[lower] + fraction * (sorted[upper] - sorted[lower]);
}

void print_summary(char const* name, std::vector<double> values) {
    std::sort(values.begin(), values.end());
    std::cout << name << ": median=" << quantile(values, 0.5) << " p95=" << quantile(values, 0.95)
              << " max=" << values.back() << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        CLI::App app{"cuDDL pairwise sketch accuracy on raw FASTA mutation cases"};

        std::string cases_path;
        std::string csv_path;
        app.add_option("--cases", cases_path, "Input cases CSV path")->required();
        app.add_option("--csv", csv_path, "Output CSV path")->required();
        CLI11_PARSE(app, argc, argv);

        std::ifstream cases(cases_path);
        if (!cases) {
            throw std::runtime_error("cannot open cases CSV: " + cases_path);
        }
        std::string line;
        if (!std::getline(cases, line)) {
            throw std::runtime_error("cases CSV is empty");
        }
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        require_header(parse_csv_line(line));

        std::ofstream csv(csv_path);
        if (!csv) {
            throw std::runtime_error("cannot open CSV output: " + csv_path);
        }
        csv << std::setprecision(17);
        csv << "implementation,generator_seed,k,buckets,power,trial,size_ratio,requested_ani,"
               "actual_ani,mutation_count,reference_bases,query_bases,reference_sha256,"
               "query_sha256,reference_path,query_path,orientation,left_cardinality,"
               "right_cardinality,intersection,lower,equal,higher,both_empty,exact_containment,"
               "sketch_containment,containment_signed_error,containment_absolute_error,"
               "exact_completeness,sketch_completeness,completeness_signed_error,"
               "completeness_absolute_error,exact_wkid,sketch_wkid,wkid_signed_error,"
               "wkid_absolute_error,exact_set_derived_ani,exact_ani,sketch_ani,"
               "ani_signed_error,ani_absolute_error\n";

        error_samples errors;
        std::optional<prepared_sequence> cached_reference;
        size_t case_count = 0;
        for (size_t row = 2; std::getline(cases, line); ++row) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty()) {
                continue;
            }
            auto const input = parse_case(parse_csv_line(line), row);

            if (!cached_reference || cached_reference->path != input.reference_path) {
                cached_reference = prepare_sequence(
                    input.reference_path, input.reference_sha256, input.reference_bases
                );
            } else if (
                cached_reference->sha256 != input.reference_sha256 ||
                cached_reference->bases != input.reference_bases
            ) {
                throw std::runtime_error(
                    "reference metadata changed for cached path: " + input.reference_path
                );
            }
            auto query = prepare_sequence(input.query_path, input.query_sha256, input.query_bases);
            auto const intersection =
                intersection_size(query.unique_kmers, cached_reference->unique_kmers);

            emit_orientation(
                csv,
                input,
                query.sketch,
                cached_reference->sketch,
                "query_to_reference",
                query.unique_kmers.size(),
                cached_reference->unique_kmers.size(),
                intersection,
                errors
            );
            if (input.size_ratio != 1U) {
                emit_orientation(
                    csv,
                    input,
                    cached_reference->sketch,
                    query.sketch,
                    "reference_to_query",
                    cached_reference->unique_kmers.size(),
                    query.unique_kmers.size(),
                    intersection,
                    errors
                );
            }
            ++case_count;
        }
        if (case_count == 0U) {
            throw std::runtime_error("cases CSV contains no data rows");
        }

        std::cout << std::setprecision(8);
        print_summary("containment absolute error", std::move(errors.containment));
        print_summary("completeness absolute error", std::move(errors.completeness));
        print_summary("WKID absolute error", std::move(errors.wkid));
        print_summary("ANI absolute error", std::move(errors.ani));
    } catch (std::exception const& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
