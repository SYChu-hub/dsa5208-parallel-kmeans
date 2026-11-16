#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Point = std::vector<double>;
using Dataset = std::vector<Point>;

struct KMeansResult {
    Dataset centroids;
    std::vector<int> labels;
    double sse = 0.0;
    int iterations = 0;
    bool converged = false;
};

double squared_distance(const Point& a, const Point& b) {
    double sum = 0.0;
    for (std::size_t d = 0; d < a.size(); ++d) {
        const double diff = a[d] - b[d];
        sum += diff * diff;
    }
    return sum;
}

void validate_data(const Dataset& data, std::size_t k,
                   int max_iterations, double tolerance) {
    if (data.empty())
        throw std::invalid_argument("Dataset must not be empty.");
    if (k == 0 || k > data.size())
        throw std::invalid_argument("k must be between 1 and n.");
    if (max_iterations <= 0)
        throw std::invalid_argument("max_iterations must be positive.");
    if (!std::isfinite(tolerance) || tolerance < 0.0)
        throw std::invalid_argument("Tolerance must be finite and non-negative.");

    const std::size_t dimensions = data.front().size();
    if (dimensions == 0)
        throw std::invalid_argument("Points must have at least one dimension.");

    for (const Point& point : data) {
        if (point.size() != dimensions)
            throw std::invalid_argument("Inconsistent point dimensions.");
        for (double value : point) {
            if (!std::isfinite(value))
                throw std::invalid_argument("Data must contain finite values.");
        }
    }
}

Dataset generate_data(std::size_t n, std::size_t dimensions,
                      unsigned int seed) {
    if (n == 0 || dimensions == 0)
        throw std::invalid_argument("n and dimensions must be positive.");

    std::mt19937 rng(seed);
    std::normal_distribution<double> distribution(0.0, 1.0);
    Dataset data(n, Point(dimensions));

    for (Point& point : data)
        for (double& value : point)
            value = distribution(rng);

    return data;
}

Dataset load_data_csv(const std::string& filename,
                      std::size_t expected_dimensions) {

    std::ifstream input_file(filename);

    if (!input_file.is_open()) {
        throw std::runtime_error(
            "Unable to open dataset file: " + filename);
    }

    Dataset data;
    std::string line;
    std::size_t line_number = 0;

    while (std::getline(input_file, line)) {
        ++line_number;

        // Ignore empty lines.
        if (line.empty())
            continue;

        std::stringstream line_stream(line);
        std::string value_text;
        Point point;

        // Read comma-separated values.
        while (std::getline(line_stream, value_text, ',')) {

            // Convert each field to a double.
            std::size_t processed = 0;
            double value;

            try {
                value = std::stod(value_text, &processed);
            }
            catch (const std::exception&) {
                throw std::runtime_error(
                    "Invalid numeric value at line " +
                    std::to_string(line_number));
            }

            // Reject unexpected characters after the number.
            if (value_text.find_first_not_of(
                    " \t\r\n", processed) != std::string::npos) {
                throw std::runtime_error(
                    "Unexpected characters at line " +
                    std::to_string(line_number));
            }

            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "Non-finite value at line " +
                    std::to_string(line_number));
            }

            point.push_back(value);
        }

        if (point.size() != expected_dimensions) {
            throw std::runtime_error(
                "Incorrect number of features at line " +
                std::to_string(line_number) +
                ". Expected " +
                std::to_string(expected_dimensions) +
                ", received " +
                std::to_string(point.size()));
        }

        data.push_back(std::move(point));
    }

    if (input_file.bad()) {
        throw std::runtime_error(
            "Error occurred while reading dataset file.");
    }

    if (data.empty()) {
        throw std::runtime_error(
            "Dataset file contains no observations.");
    }

    std::cout << "Loaded dataset: " << filename << '\n'
              << "Observations: " << data.size() << '\n'
              << "Dimensions: " << expected_dimensions << '\n';

    return data;
}

Dataset initialize_centroids(const Dataset& data, std::size_t k,
                             unsigned int seed) {
    std::vector<std::size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    Dataset centroids;
    centroids.reserve(k);

    for (std::size_t i = 0; i < k; ++i)
        centroids.push_back(data[indices[i]]);

    return centroids;
}

int nearest_centroid(const Point& point, const Dataset& centroids) {
    int best_cluster = 0;
    double best_distance = squared_distance(point, centroids[0]);

    for (std::size_t j = 1; j < centroids.size(); ++j) {
        const double distance = squared_distance(point, centroids[j]);
        if (distance < best_distance) {
            best_distance = distance;
            best_cluster = static_cast<int>(j);
        }
    }

    return best_cluster;
}

KMeansResult kmeans(const Dataset& data, std::size_t k,
                    int max_iterations, double tolerance,
                    unsigned int initialization_seed) {
    validate_data(data, k, max_iterations, tolerance);

    const std::size_t n = data.size();
    const std::size_t dimensions = data[0].size();

    KMeansResult result;
    result.centroids = initialize_centroids(data, k, initialization_seed);
    result.labels.assign(n, -1);

    Dataset sums(k, Point(dimensions, 0.0));
    std::vector<std::size_t> counts(k, 0);

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        std::fill(counts.begin(), counts.end(), 0);
        for (Point& sum : sums)
            std::fill(sum.begin(), sum.end(), 0.0);

        // Assign points to the nearest centroid and accumulate sums.
        for (std::size_t i = 0; i < n; ++i) {
            const int cluster = nearest_centroid(data[i], result.centroids);
            const std::size_t c = static_cast<std::size_t>(cluster);

            result.labels[i] = cluster;
            ++counts[c];

            for (std::size_t d = 0; d < dimensions; ++d)
                sums[c][d] += data[i][d];
        }

        // Update centroids and measure the largest movement.
        double max_movement_squared = 0.0;

        for (std::size_t j = 0; j < k; ++j) {
            // Empty-cluster policy: retain the previous centroid.
            if (counts[j] == 0)
                continue;

            Point updated(dimensions);
            for (std::size_t d = 0; d < dimensions; ++d)
                updated[d] = sums[j][d] / static_cast<double>(counts[j]);

            max_movement_squared = std::max(
                max_movement_squared,
                squared_distance(result.centroids[j], updated));

            result.centroids[j] = std::move(updated);
        }

        result.iterations = iteration + 1;

        if (max_movement_squared <= tolerance * tolerance) {
            result.converged = true;
            break;
        }
    }

    // Reassign using final centroids, then calculate final SSE.
    result.sse = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        const int cluster = nearest_centroid(data[i], result.centroids);
        result.labels[i] = cluster;
        result.sse += squared_distance(
            data[i], result.centroids[static_cast<std::size_t>(cluster)]);
    }

    if (!std::isfinite(result.sse))
        throw std::overflow_error("SSE is not finite.");

    return result;
}



#ifndef KMEANS_NO_MAIN
int main(int argc, char** argv) {
    std::size_t n = 10000;
    std::size_t dimensions = 8;
    std::size_t k = 8;

    constexpr unsigned int data_seed = 42;
    constexpr unsigned int initialization_seed = 123;
    constexpr int max_iterations = 500;
    constexpr double tolerance = 1e-4;

    std::string dataset_path;
    bool use_real_data = false;

    try {
        // Parse command-line arguments.
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];

            if (arg == "--data" && i + 1 < argc) {
                use_real_data = true;
                dataset_path = argv[++i];
            } else if (arg == "--n" && i + 1 < argc) {
                n = std::stoull(argv[++i]);
            } else if (arg == "--dimensions" && i + 1 < argc) {
                dimensions = std::stoull(argv[++i]);
            } else if (arg == "--k" && i + 1 < argc) {
                k = std::stoull(argv[++i]);
            } else {
                throw std::invalid_argument(
                    "Unknown or incomplete argument: " + arg);
            }
        }

        if (n == 0 || dimensions == 0 || k == 0) {
            throw std::invalid_argument(
                "n, dimensions and k must be positive.");
        }

        if (use_real_data) {
            // Covertype uses 55 dimensions and 7 clusters.
            dimensions = 55;
            k = 7;
        }

        Dataset data;

        if (use_real_data) {
            data = load_data_csv(dataset_path, dimensions);
            n = data.size();
        } else {
            data = generate_data(n, dimensions, data_seed);
        }

        validate_data(data, k, max_iterations, tolerance);

        std::cout
            << "Algorithm: Serial K-means Benchmark\n"
            << "Dataset: "
            << (use_real_data ? "Covertype" : "Synthetic") << '\n'
            << "Points: " << n << '\n'
            << "Dimensions: " << dimensions << '\n'
            << "Clusters: " << k << '\n'
            << "Data seed: " << data_seed << '\n'
            << "Initialization seed: " << initialization_seed << '\n';

        const auto start_time =
            std::chrono::steady_clock::now();

        const KMeansResult result = kmeans(
            data,
            k,
            max_iterations,
            tolerance,
            initialization_seed);

        const auto end_time =
            std::chrono::steady_clock::now();

        const double runtime_seconds =
            std::chrono::duration<double>(
                end_time - start_time).count();

        std::cout << std::setprecision(15)
                  << "Iterations: " << result.iterations << '\n'
                  << "Converged: "
                  << (result.converged ? "yes" : "no") << '\n'
                  << "SSE: " << result.sse << '\n'
                  << "Runtime (seconds): "
                  << runtime_seconds << '\n';

        for (std::size_t j = 0;
             j < result.centroids.size(); ++j) {
            std::cout << "Centroid " << j << ':';

            for (double value : result.centroids[j]) {
                std::cout << ' ' << value;
            }

            std::cout << '\n';
        }

    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
#endif
