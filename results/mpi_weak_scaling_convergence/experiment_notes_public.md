# Consolidated MPI K-means Weak-Scaling Results

## Scope

This package consolidates the final 1, 2, 4, 8, 16 and 32 MPI-rank
weak-scaling measurements. It contains 30 measured runs: five repetitions
for each configuration, excluding one warm-up per configuration.

## Experimental configuration

- Algorithm: distributed MPI K-means.
- Synthetic observations: 10,000 per MPI rank.
- Feature dimensions: 8.
- Number of clusters: 8.
- Random data seed: 42.
- Centroid seed: 123.
- Maximum iterations: 3,000.
- Convergence tolerance: 0.0001.
- Executable SHA256:
  4ebe70fe6462e3a5aaabfa1046dc2b690730667f3ceaf21a693677a86c56d23a

Both source experiment manifests identify the same final executable.

## Timing definitions

`computation_seconds` measures the computation phase after data
distribution/reconstruction.

`parallel_execution_seconds` includes the wider parallel execution,
including data distribution/reconstruction and final SSE reduction/barrier.
Use parallel execution time as the primary end-to-end weak-scaling measure.

For each configuration, the summary reports the mean, median, sample
standard deviation and coefficient of variation (CV) across five runs.
The median is used for the primary scaling comparison.

## Observed weak-scaling efficiency

The reported empirical ratio is:

    E_observed(p) = median(T_parallel, 1 rank) /
                    median(T_parallel, p ranks)

This is an observed timing ratio, not a pure MPI efficiency measure.

## Important interpretation limitation

The iteration counts differ across configurations. In particular, they
are 83, 196, 257, 728, 899 and 718 for 1, 2, 4, 8, 16 and 32 ranks,
respectively. Consequently, total K-means algorithmic work is not constant
per rank across these experiments.

The measured timing and efficiency therefore reflect both runtime scaling
and differences in convergence work. They must not be interpreted as an
isolated measurement of MPI communication overhead.

All measured runs reported convergence. Within each rank configuration,
the five runs have identical iteration counts and SSE.

## Provenance

`provenance/` contains copies of the two original raw CSVs and their
SHA256 manifests. The original source experiment directories are not
modified by this script.

Source experiments:
- `results/mpi_weak_scaling_final_124_20261107`

## Files

- `weak_scaling_consolidated_raw.csv`: all 30 measurements.
- `weak_scaling_consolidated_summary.csv`: six-row statistical summary.
- `weak_scaling_parallel_execution.svg`: median parallel execution time.
- `weak_scaling_computation_time.svg`: median computation time.
- `weak_scaling_observed_efficiency.svg`: empirical efficiency ratio.
- `provenance/`: original source CSVs and checksum manifests.
