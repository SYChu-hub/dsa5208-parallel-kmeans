#define KMEANS_NO_MAIN
#include "kmeans_serial_bench.cpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <climits>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <string>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    try {
        // ---------------------------------------------------------
        // 1. Configure the dataset.
        // ---------------------------------------------------------
        // ---------------------------------------------------------
        std::size_t n = 10000;
        std::size_t dimensions = 8;
        std::size_t k = 8;

        constexpr unsigned int data_seed = 42;
        constexpr unsigned int initialization_seed = 123;

        constexpr int max_iterations = 3000;
        constexpr double tolerance = 1e-4;
        int fixed_iterations = 0;

        bool use_real_data = false;
        std::string dataset_path;

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
            } else if (arg == "--fixed-iterations" && i + 1 < argc) {
                fixed_iterations = std::stoi(argv[++i]);
            } else {
                throw std::invalid_argument(
                    "Unknown or incomplete argument: " + arg);
            }
        }

        if (n == 0 || dimensions == 0 || k == 0) {
            throw std::invalid_argument(
                "n, dimensions and k must be positive.");
        }
        if (fixed_iterations < 0) {
            throw std::invalid_argument(
                "--fixed-iterations must be positive when supplied.");
        }
        const int iteration_limit =
            fixed_iterations > 0 ? fixed_iterations : max_iterations;

        if (use_real_data) {
            dimensions = 55;
            k = 7;
        }

        // ---------------------------------------------------------
        // 2. Rank 0 loads or generates the dataset.
        // Dataset preparation is excluded from benchmark timing.
        // ---------------------------------------------------------
        Dataset data;
        std::vector<double> global_flat_data;

        int data_ready = 1;
        std::string data_error;

        if (rank == 0) {
            try {
                if (use_real_data) {
                    data = load_data_csv(dataset_path, dimensions);
                    n = data.size();
                } else {
                    data = generate_data(n, dimensions, data_seed);
                }

                validate_data(data, k, iteration_limit, tolerance);

            }
            catch (const std::exception& error) {
                data_ready = 0;
                data_error = error.what();
            }
        }

        // Inform every rank whether dataset preparation succeeded.
        MPI_Bcast(
            &data_ready,
            1,
            MPI_INT,
            0,
            MPI_COMM_WORLD);

        // Stop all ranks together if rank 0 failed to prepare the dataset.
        if (!data_ready) {
            if (rank == 0) {
                std::cerr << "Dataset preparation failed: "
                          << data_error << '\n';
            }

            MPI_Finalize();
            return 1;
        }

        // Broadcast the actual observation count to every rank.
        unsigned long long n_broadcast =
            static_cast<unsigned long long>(n);

        MPI_Bcast(
            &n_broadcast,
            1,
            MPI_UNSIGNED_LONG_LONG,
            0,
            MPI_COMM_WORLD);

        n = static_cast<std::size_t>(n_broadcast);

        // Ensure MPI's int-based counts and displacements are safe.
        if (n > static_cast<std::size_t>(INT_MAX) / dimensions) {
            throw std::overflow_error(
                "Dataset is too large for MPI int counts.");
        }

        // ---------------------------------------------------------
        // 3. Partition the dataset across ranks.
        // The first remainder ranks receive one extra point.
        // ---------------------------------------------------------
        const std::size_t base_points =
            n / static_cast<std::size_t>(world_size);

        const std::size_t remainder =
            n % static_cast<std::size_t>(world_size);

        std::vector<int> value_counts(world_size);
        std::vector<int> value_displacements(world_size);

        int value_offset = 0;

        for (int r = 0; r < world_size; ++r) {
            const std::size_t rank_points =
                base_points +
                (static_cast<std::size_t>(r) < remainder ? 1 : 0);

            const std::size_t rank_values =
                rank_points * dimensions;

            value_counts[r] = static_cast<int>(rank_values);
            value_displacements[r] = value_offset;

            value_offset += value_counts[r];
        }

        const std::size_t local_n =
            base_points +
            (static_cast<std::size_t>(rank) < remainder ? 1 : 0);

        // Rank 0 flattens the dataset for MPI_Scatterv.
        if (rank == 0) {

            global_flat_data.reserve(n * dimensions);

            for (const Point& point : data) {
                global_flat_data.insert(
                    global_flat_data.end(),
                    point.begin(),
                    point.end());
            }
        }

        // ---------------------------------------------------------
        // 3. Start timing before MPI data distribution.
        // Synthetic data generation is excluded; distribution,
        // centroid initialization and K-means execution are timed.
        // ---------------------------------------------------------
        MPI_Barrier(MPI_COMM_WORLD);
        const double start_time = MPI_Wtime();

        // Distribute the flattened point coordinates.
        const int local_value_count = value_counts[rank];

        std::vector<double> local_flat_data(
            static_cast<std::size_t>(local_value_count));

        MPI_Scatterv(
            rank == 0 ? global_flat_data.data() : nullptr,
            value_counts.data(),
            value_displacements.data(),
            MPI_DOUBLE,
            local_flat_data.data(),
            local_value_count,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD);

        // Reconstruct the local dataset on each rank.
        Dataset local_data(local_n, Point(dimensions));

        for (std::size_t i = 0; i < local_n; ++i) {
            for (std::size_t d = 0; d < dimensions; ++d) {
                local_data[i][d] =
                    local_flat_data[i * dimensions + d];
            }
        }


        // Start computation timing after data distribution
        // and local-data reconstruction.
        MPI_Barrier(MPI_COMM_WORLD);
        const double computation_start_time = MPI_Wtime();

        Dataset centroids;

        if (rank == 0) {
            centroids = initialize_centroids(
                data, k, initialization_seed);
        }

        std::vector<double> flat_centroids(k * dimensions, 0.0);

        if (rank == 0) {
            for (std::size_t c = 0; c < k; ++c) {
                for (std::size_t d = 0; d < dimensions; ++d) {
                    flat_centroids[c * dimensions + d] =
                        centroids[c][d];
                }
            }
        }

        MPI_Bcast(
            flat_centroids.data(),
            static_cast<int>(flat_centroids.size()),
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD);

        centroids.assign(k, Point(dimensions));

        for (std::size_t c = 0; c < k; ++c) {
            for (std::size_t d = 0; d < dimensions; ++d) {
                centroids[c][d] =
                    flat_centroids[c * dimensions + d];
            }
        }

        // ---------------------------------------------------------
        // 4. Allocate local/global cluster statistics.
        // ---------------------------------------------------------
        std::vector<double> local_sums(k * dimensions, 0.0);
        std::vector<double> global_sums(k * dimensions, 0.0);

        std::vector<unsigned long long> local_counts(k, 0);
        std::vector<unsigned long long> global_counts(k, 0);

        int iterations = 0;
        bool converged = false;

        // ---------------------------------------------------------
        // 5. Iterative distributed K-means.
        // ---------------------------------------------------------
        for (int iteration = 0;
             iteration < iteration_limit;
             ++iteration) {

            std::fill(local_sums.begin(), local_sums.end(), 0.0);
            std::fill(local_counts.begin(), local_counts.end(), 0);

            // Assign local points to their nearest centroids.
            for (const Point& point : local_data) {
                const std::size_t cluster =
                    nearest_centroid(point, centroids);

                ++local_counts[cluster];

                for (std::size_t d = 0; d < dimensions; ++d) {
                    local_sums[cluster * dimensions + d] += point[d];
                }
            }

            // Combine coordinate sums across all MPI ranks.
            MPI_Allreduce(
                local_sums.data(),
                global_sums.data(),
                static_cast<int>(global_sums.size()),
                MPI_DOUBLE,
                MPI_SUM,
                MPI_COMM_WORLD);

            // Combine cluster counts across all MPI ranks.
            MPI_Allreduce(
                local_counts.data(),
                global_counts.data(),
                static_cast<int>(global_counts.size()),
                MPI_UNSIGNED_LONG_LONG,
                MPI_SUM,
                MPI_COMM_WORLD);

            // Update centroids using the global statistics.
            double max_movement_squared = 0.0;

            for (std::size_t c = 0; c < k; ++c) {
                // Retain the previous centroid for an empty cluster.
                if (global_counts[c] == 0) {
                    continue;
                }

                Point updated(dimensions);

                for (std::size_t d = 0; d < dimensions; ++d) {
                    updated[d] =
                        global_sums[c * dimensions + d] /
                        static_cast<double>(global_counts[c]);
                }

                max_movement_squared = std::max(
                    max_movement_squared,
                    squared_distance(centroids[c], updated));

                centroids[c] = std::move(updated);
            }

            // All ranks make the same convergence decision.
            double global_max_movement_squared = 0.0;

            MPI_Allreduce(
                &max_movement_squared,
                &global_max_movement_squared,
                1,
                MPI_DOUBLE,
                MPI_MAX,
                MPI_COMM_WORLD);

            iterations = iteration + 1;

            if (global_max_movement_squared <=
                tolerance * tolerance) {
                converged = true;
                if (fixed_iterations == 0) {
                    break;
                }
            }
        }

        // ---------------------------------------------------------
        // 6. Calculate final SSE using the final centroids.
        // ---------------------------------------------------------
        double local_sse = 0.0;

        for (const Point& point : local_data) {
            const std::size_t cluster =
                nearest_centroid(point, centroids);

            local_sse += squared_distance(
                point, centroids[cluster]);
        }

        double global_sse = 0.0;

        MPI_Allreduce(
            &local_sse,
            &global_sse,
            1,
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD);

        // Synchronize completion before recording elapsed time.
        MPI_Barrier(MPI_COMM_WORLD);

        const double end_time = MPI_Wtime();

        // Record elapsed times independently on every rank.
        const double local_computation_seconds =
            end_time - computation_start_time;

        const double local_parallel_execution_seconds =
            end_time - start_time;

        // Use the slowest rank's elapsed time.
        double max_computation_seconds = 0.0;
        double max_parallel_execution_seconds = 0.0;

        MPI_Reduce(
            &local_computation_seconds,
            &max_computation_seconds,
            1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        MPI_Reduce(
            &local_parallel_execution_seconds,
            &max_parallel_execution_seconds,
            1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (!std::isfinite(global_sse)) {
            throw std::overflow_error("SSE is not finite.");
        }

        // ---------------------------------------------------------
        // 7. Print results once, from rank 0.
        // ---------------------------------------------------------
        if (rank == 0) {
            std::cout << std::setprecision(15)
                      << "Algorithm: MPI K-means\n"
                      << "MPI processes: " << world_size << '\n'
                      << "Dataset: "
                      << (use_real_data ? "Covertype" : "Synthetic")
                      << '\n'
                      << "Points: " << n << '\n'
                      << "Dimensions: " << dimensions << '\n'
                      << "Clusters: " << k << '\n'
                      << "Execution mode: "
                      << (fixed_iterations > 0 ? "fixed_iterations" : "convergence_based")
                      << '\n'
                      << "Iterations: " << iterations << '\n';

            if (fixed_iterations > 0) {
                std::cout
                    << "Convergence criterion reached during fixed run: "
                    << (converged ? "yes" : "no") << '\n';
            } else {
                std::cout
                    << "Converged: "
                    << (converged ? "yes" : "no") << '\n';
            }

            std::cout
                      << "SSE: " << global_sse << '\n'
                      << "Computation time (seconds): "
                      << max_computation_seconds << '\n'
                      << "Parallel execution time (seconds): "
                      << max_parallel_execution_seconds << '\n';

            std::cout << "Centroids:\n";

            for (std::size_t c = 0; c < centroids.size(); ++c) {
                std::cout << "  Cluster " << c << ':';

                for (double value : centroids[c]) {
                    std::cout << ' ' << value;
                }

                std::cout << '\n';
            }
        }

    } catch (const std::exception& error) {
        std::cerr << "Error on MPI rank " << rank
                  << ": " << error.what() << '\n';

        // Stop all ranks if one rank encounters an error.
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    MPI_Finalize();
    return 0;
}
