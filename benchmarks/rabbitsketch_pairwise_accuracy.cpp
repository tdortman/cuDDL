#include <api/SketchBuilder.h>
#include <api/Version.h>
#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

using json = nlohmann::json;
namespace api = Sketch::API;

int main(int argc, char** argv) try {
    CLI::App app{"Replay raw-DNA accuracy cases with RabbitSketch FastKMV"};
    std::string input, output;
    app.add_option("--cases", input, "cuDDL accuracy JSON containing exact set metrics")
        ->required();
    app.add_option("--output", output, "RabbitSketch accuracy JSON")->required();
    CLI11_PARSE(app, argc, argv);

    std::ifstream source(input);
    auto report = json::parse(source);
    if (report.at("schema") != "cuddl-benchmark/v1" ||
        report.at("operation") != "pairwise_accuracy" || report.at("measurements").empty()) {
        throw std::runtime_error("expected a nonempty cuDDL pairwise accuracy report");
    }
    api::SketchConfig config;
    config.algorithm = api::Algorithm::FastKMV;
    config.sampling = api::SamplingMode::BottomK;
    config.aggregation = api::AggregationMode::OneSketchPerFile;
    config.ambiguous_policy = api::AmbiguousPolicy::SkipKmer;
    config.kmer_size = 25;
    config.resolution = 2048;
    config.seed = 42;
    config.canonical = true;
    config.validate();
    std::map<std::string, api::BuildResult> sketches;
    auto load = [&](std::string const& path) -> api::BuildResult const& {
        if (!sketches.contains(path)) {
            auto built = api::buildFastxFiles({path}, {config});
            if (built.size() != 1) throw std::runtime_error("expected one sketch for " + path);
            sketches.emplace(path, std::move(built.front()));
        }
        return sketches.at(path);
    };
    for (auto& row : report.at("measurements")) {
        auto const& task = row.at("case");
        auto const truth = row.at("metrics");
        if (row.at("implementation").at("name") != "cuddl" || task.at("k") != 25 ||
            task.at("buckets") != 2048) {
            throw std::runtime_error("expected cuDDL cases with k=25 and 2048 buckets");
        }
        auto const orientation = truth.at("orientation").get<std::string>();
        bool const forward = orientation == "query_to_reference";
        if (!forward && orientation != "reference_to_query") {
            throw std::runtime_error("unknown pair orientation: " + orientation);
        }
        auto const& left = load(task.at(forward ? "query_path" : "reference_path"));
        auto const& right = load(task.at(forward ? "reference_path" : "query_path"));
        auto const result = left.sketch.query(right.sketch);
        if (!result.estimate_available || !(result.left_cardinality > 0.0) ||
            !(result.right_cardinality > 0.0)) {
            throw std::runtime_error("RabbitSketch estimates unavailable: " + result.plan.reason);
        }
        json metrics;
        for (auto const* key :
             {"orientation",
              "left_cardinality",
              "right_cardinality",
              "intersection",
              "exact_set_derived_ani"}) {
            metrics[key] = truth.at(key);
        }
        auto add = [&](std::string const& name, double estimate) {
            if (!std::isfinite(estimate)) {
                throw std::runtime_error("RabbitSketch returned nonfinite " + name);
            }
            double const exact = truth.at("exact_" + name);
            metrics["exact_" + name] = exact;
            metrics["sketch_" + name] = estimate;
            metrics[name + "_signed_error"] = estimate - exact;
            metrics[name + "_absolute_error"] = std::abs(estimate - exact);
        };
        add("cardinality", result.left_cardinality);
        add("containment", result.left_containment);
        add("completeness", std::min(1.0, result.left_cardinality / result.right_cardinality));
        add("wkid", result.max_containment);
        add("ani", result.ani);
        metrics["cardinality_relative_error"] = metrics["cardinality_signed_error"].get<double>() /
                                                truth.at("exact_cardinality").get<double>();
        metrics["cardinality_absolute_relative_error"] =
            std::abs(metrics["cardinality_relative_error"].get<double>());
        metrics["sketch_containment_ani"] = result.max_containment_ani;
        metrics["containment_ani_absolute_error"] =
            std::abs(result.max_containment_ani - truth.at("exact_ani").get<double>());
        metrics["sketch_jaccard"] = result.jaccard;
        metrics["ani_estimator"] = "Jaccard: (2J/(1+J))^(1/k)";
        metrics["wkid_estimator"] = "max(left_containment, right_containment)";
        metrics["completeness_estimator"] = "min(1, estimated_left/estimated_right)";
        metrics["query_method"] = Sketch::Query::methodName(result.plan.method);
        row["metrics"] = std::move(metrics);
        row["case"]["hash_seed"] = config.seed;
        row["case"]["sketch_size"] = config.resolution;
        row["implementation"] = {
            {"name", "rabbitsketch"},
            {"variant", "FastKMV"},
            {"version", api::VERSION},
            {"revision", RABBITSKETCH_REVISION}
        };
    }
    std::ofstream destination(output);
    destination << report.dump(2) << '\n';
    destination.close();
    if (!destination) throw std::runtime_error("failed to write " + output);
    std::cout << "Saved " << report["measurements"].size() << " RabbitSketch accuracy rows\n";
} catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
