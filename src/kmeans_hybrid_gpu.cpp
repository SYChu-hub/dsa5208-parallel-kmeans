
#define KMEANS_NO_MAIN
#include "kmeans_serial.cpp"

#include <mpi.h>
#include <omp.h>

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
        std::size_t n = 10000;
        std::size_t dimensions = 8;
        std::size_t k = 8;

        constexpr unsigned int data_seed = 42;
        constexpr unsigned int initialization_seed = 123;

        constexpr int max_iterations = 500;
        constexpr double tolerance = 1e-4;

        bool use_real_data = false;
        std::string dataset_path;

        if (argc == 3 &&
            std::string(argv[1]) == "--data") {

            use_real_data = true;
            dataset_path = argv[2];

            dimensions = 55;
            k = 7;

        } else if (argc != 1) {

            throw std::invalid_argument(
                "Usage: mpirun -np <ranks> ./kmeans_hybrid "
                "[--data <csv_file>]");
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

                validate_data(data, k, max_iterations, tolerance);

            } catch (const std::exception& error) {
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

        // Allocate per-thread cluster statistics once, not every iteration.
        const std::size_t sum_stride = k * dimensions;

        const int thread_count = omp_get_max_threads();

        int actual_thread_count = 0;
        int expected_thread_count = 0;

        std::vector<double> thread_sums(
            static_cast<std::size_t>(thread_count) * sum_stride,
            0.0);

        std::vector<unsigned long long> thread_counts(
            static_cast<std::size_t>(thread_count) * k,
            0);

        // GPU-produced nearest-centroid labels, one per local observation.
        std::vector<int> local_labels(local_n, 0);

        // Use raw pointers in OpenMP target regions.
        double* gpu_data = local_flat_data.data();
        double* gpu_centroids = flat_centroids.data();
        int* gpu_labels = local_labels.data();

        // Confirm that OpenMP target execution is using an accelerator.
        int target_is_initial_device = 1;

        #pragma omp target map(from: target_is_initial_device)
        {
            target_is_initial_device = omp_is_initial_device();
        }

        if (rank == 0) {
            std::cout << "OpenMP devices available: "
                      << omp_get_num_devices() << std::endl;
            std::cout << "K-means target executing on "
                      << (target_is_initial_device
                              ? "initial CPU device"
                              : "GPU device")
                      << std::endl;
        }

        if (target_is_initial_device != 0) {
            if (rank == 0) {
                std::cerr
                    << "ERROR: OpenMP target is running on the initial "
                    << "CPU device; GPU execution is required."
                    << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        int iterations = 0;
        bool converged = false;

        // ---------------------------------------------------------
        // 5. Iterative distributed K-means.
        // ---------------------------------------------------------
        for (int iteration = 0;
             iteration < max_iterations;
             ++iteration) {

            // Record the actual team size for this iteration.
            int iteration_thread_count = 0;

            // Reset this rank's statistics and all thread-local statistics.
            std::fill(local_sums.begin(), local_sums.end(), 0.0);
            std::fill(local_counts.begin(), local_counts.end(), 0);
            std::fill(thread_sums.begin(), thread_sums.end(), 0.0);
            std::fill(thread_counts.begin(), thread_counts.end(), 0);

            // Refresh flattened centroids after the preceding update.
            for (std::size_t c = 0; c < k; ++c) {
                for (std::size_t d = 0; d < dimensions; ++d) {
                    flat_centroids[c * dimensions + d] = centroids[c][d];
                }
            }

            // GPU assignment: find the nearest centroid for each local point.
            #pragma omp target teams distribute parallel for \
                map(to: gpu_data[0:local_value_count], \
                        gpu_centroids[0:sum_stride]) \
                map(from: gpu_labels[0:local_n])
            for (std::size_t i = 0; i < local_n; ++i) {
                int best_cluster = 0;
                double best_distance = 0.0;

                for (std::size_t d = 0; d < dimensions; ++d) {
                    const double diff =
                        gpu_data[i * dimensions + d] - gpu_centroids[d];
                    best_distance += diff * diff;
                }

                for (std::size_t c = 1; c < k; ++c) {
                    double distance = 0.0;

                    for (std::size_t d = 0; d < dimensions; ++d) {
                        const double diff =
                            gpu_data[i * dimensions + d] -
                            gpu_centroids[c * dimensions + d];
                        distance += diff * diff;
                    }

                    if (distance < best_distance) {
                        best_distance = distance;
                        best_cluster = static_cast<int>(c);
                    }
                }

                gpu_labels[i] = best_cluster;
            }

            // Validate GPU labels before CPU accumulation.
            for (std::size_t i = 0; i < local_n; ++i) {
                if (local_labels[i] < 0 ||
                    static_cast<std::size_t>(local_labels[i]) >= k) {
                    std::cerr << "Rank " << rank
                              << ": invalid GPU cluster label at local point "
                              << i << "." << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 3);
                }
            }

            // CPU OpenMP accumulation using the GPU-produced labels.
            #pragma omp parallel num_threads(thread_count)
            {
                #pragma omp single
                {
                    iteration_thread_count = omp_get_num_threads();
                }

                const int tid = omp_get_thread_num();

                const std::size_t sum_offset =
                    static_cast<std::size_t>(tid) * sum_stride;

                const std::size_t count_offset =
                    static_cast<std::size_t>(tid) * k;

                #pragma omp for schedule(static)
                for (std::size_t i = 0; i < local_data.size(); ++i) {
                    const Point& point = local_data[i];

                    const std::size_t cluster =
                        static_cast<std::size_t>(local_labels[i]);

                    ++thread_counts[count_offset + cluster];

                    for (std::size_t d = 0; d < dimensions; ++d) {
                        thread_sums[
                            sum_offset + cluster * dimensions + d
                        ] += point[d];
                    }
                }
            }

            // -----------------------------------------------------
            // 5a. Verify OpenMP team-size consistency.
            // -----------------------------------------------------
            int min_thread_count = 0;
            int max_thread_count = 0;

            // Find the minimum actual thread count across MPI ranks.
            MPI_Allreduce(
                &iteration_thread_count,
                &min_thread_count,
                1,
                MPI_INT,
                MPI_MIN,
                MPI_COMM_WORLD);

            // Find the maximum actual thread count across MPI ranks.
            MPI_Allreduce(
                &iteration_thread_count,
                &max_thread_count,
                1,
                MPI_INT,
                MPI_MAX,
                MPI_COMM_WORLD);

            // Establish the expected team size during the first iteration.
            if (iteration == 0) {
                expected_thread_count = min_thread_count;
            }

            // Detect differences between ranks or across iterations.
            if (min_thread_count != max_thread_count ||
                iteration_thread_count != expected_thread_count) {

                if (rank == 0) {
                    std::cerr
                        << "OpenMP thread-count inconsistency detected "
                        << "at iteration " << iteration + 1 << ".\n"
                        << "Expected threads per MPI rank: "
                        << expected_thread_count << '\n'
                        << "Minimum actual threads across ranks: "
                        << min_thread_count << '\n'
                        << "Maximum actual threads across ranks: "
                        << max_thread_count << '\n';
                }

                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Record the verified actual team size.
            actual_thread_count = iteration_thread_count;

            // -----------------------------------------------------
            // 5b. Merge thread-local statistics into rank totals.
            // -----------------------------------------------------
            for (int t = 0; t < iteration_thread_count; ++t) {
                const std::size_t sum_offset =
                    static_cast<std::size_t>(t) * sum_stride;

                const std::size_t count_offset =
                    static_cast<std::size_t>(t) * k;

                for (std::size_t c = 0; c < k; ++c) {
                    local_counts[c] += thread_counts[count_offset + c];

                    for (std::size_t d = 0; d < dimensions; ++d) {
                        local_sums[c * dimensions + d] +=
                            thread_sums[
                                sum_offset + c * dimensions + d
                            ];
                    }
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
                break;
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

        const double runtime_seconds =
            MPI_Wtime() - start_time;

        if (!std::isfinite(global_sse)) {
            throw std::overflow_error("SSE is not finite.");
        }

        // ---------------------------------------------------------
        // 7. Print results once, from rank 0.
        // ---------------------------------------------------------
        if (rank == 0) {
            std::cout << std::setprecision(15)
                      << "Algorithm: Hybrid MPI + OpenMP K-means\n"
                      << "MPI processes: " << world_size << '\n'
                      << "OpenMP threads per process: "
                      << actual_thread_count << '\n'
                      << "Total OpenMP threads: "
                      << world_size * actual_thread_count << '\n'
                      << "Dataset: "
                      << (use_real_data ? "Covertype" : "Synthetic")
                      << '\n'
                      << "Points: " << n << '\n'
                      << "Dimensions: " << dimensions << '\n'
                      << "Clusters: " << k << '\n'
                      << "Iterations: " << iterations << '\n'
                      << "Converged: "
                      << (converged ? "yes" : "no") << '\n'
                      << "SSE: " << global_sse << '\n'
                      << "Runtime seconds: "
                      << runtime_seconds << '\n';

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
