# Parallel K-Means on NUS Vanda HPC

This repository contains selected source code, benchmark measurements,
experiment notes, and figures for the DSA5208 parallel K-means project.
The project report is submitted separately and is not included here.

## Project overview

The project implements and evaluates K-means clustering using:

- Serial C++.
- Distributed-memory parallelism with MPI.
- Hybrid MPI and OpenMP parallelism.
- GPU acceleration using OpenMP target offloading.

The experiments examine runtime, speedup, parallel efficiency, convergence,
communication, synchronisation, and data movement.

The principal real-world dataset is Forest CoverType:
581,012 observations, 55 features, and 7 clusters.
Synthetic datasets are used for MPI weak-scaling experiments.

## Repository contents

- `src/`: serial, MPI, hybrid, and GPU implementation source files.
- `scripts/`: dataset preparation and inspection scripts.
- `results/`: benchmark CSV files, experiment notes, and summaries.
- `results/report_figures/`: ten SVG benchmark figures.

The repository excludes raw datasets, compiled executables, PBS scripts,
scheduler logs, node allocation files, and the separately submitted report.

## Implementations

### Serial C++

- `src/kmeans_serial.cpp`: serial K-means implementation.
- `src/benchmark/kmeans_serial_fair.cpp`: controlled serial benchmark.
- `src/benchmark/kmeans_serial_bench.cpp`: common benchmark implementation
  used by parallel benchmark sources.

### MPI

- `src/benchmark/kmeans_mpi_bench.cpp`: available MPI benchmark source.
- `src/benchmark/kmeans_mpi_bench_fixed_weak.cpp`: source snapshot
  associated with the fixed-iteration weak-scaling experiment.

MPI distributes observations across ranks and uses collective operations
to combine cluster sums and counts and coordinate convergence.

### Hybrid MPI and OpenMP

`src/benchmark/kmeans_hybrid_controlled.cpp` is the source snapshot
associated with the controlled full-dataset hybrid benchmark.

### GPU

`src/kmeans_hybrid_gpu.cpp` uses OpenMP target offloading to execute work
on an NVIDIA GPU. It is not a custom CUDA-kernel implementation.

## Benchmark results

### Serial problem-size experiment

Five measured runs were collected per dataset size. The table reports
median runtime.

| Observations | Median runtime (s) | Iterations |
|---:|---:|---:|
| 50,000 | 0.719656 | 51 |
| 100,000 | 3.537158 | 189 |
| 200,000 | 1.941394 | 50 |
| 300,000 | 3.083305 | 52 |
| 400,000 | 7.137724 | 95 |
| 500,000 | 8.650530 | 91 |
| 581,012 | 9.344262 | 88 |

Iteration counts differ, so runtime reflects both dataset size and
convergence work. These results are not a pure linear scaling experiment.

Data: `results/serial_problem_scaling/`
The public CSV is `serial_scaling_runs_public.csv`; the original local
log-path column is excluded from the published copy.
Figure: `results/report_figures/fig01_serial_scaling.svg`

### MPI strong scaling

The primary controlled experiment used the full Covertype dataset,
K=7, one warm-up and three measured runs per configuration.

| MPI ranks | Median parallel execution (s) | Speedup | Efficiency |
|---:|---:|---:|---:|
| 8 | 1.195044 | 1.000 | 100.0% |
| 16 | 0.619475 | 1.929 | 96.5% |
| 32 | 0.374059 | 3.195 | 79.9% |
| 64 | 0.325662 | 3.670 | 45.9% |

All measured runs converged in 88 iterations and reported consistent SSE.
The 8-, 16-, and 32-rank configurations used one node; 64 ranks used
two nodes. Speedup and efficiency use 8 ranks as the baseline.

The 128-rank diagnostic and separate 256-rank controlled experiment are
provided as supplementary data, not as part of the primary 8-to-64-rank
comparison. The historical MPI8/MPI256 comparison is also supplementary
and contains one measured observation per configuration.

Data: `results/mpi_strong_scaling/`
Figures:
- `results/report_figures/fig02_mpi_strong_runtime.svg`
- `results/report_figures/fig03_mpi_strong_speedup.svg`

### MPI weak scaling: convergence-driven experiment

This experiment uses synthetic data with 10,000 observations per rank,
8 features, K=8, and five measured runs per configuration after warm-up.
It evaluates 1, 2, 4, 8, 16, and 32 MPI ranks.

| MPI ranks | Iterations |
|---:|---:|
| 1 | 83 |
| 2 | 196 |
| 4 | 257 |
| 8 | 728 |
| 16 | 899 |
| 32 | 718 |

All measured runs converged. However, iteration counts differ, so total
algorithmic work is not constant per rank. The observed timing ratio is
an empirical comparison, not a pure measure of MPI communication
efficiency.

Data: `results/mpi_weak_scaling_convergence/`
Notes: `results/mpi_weak_scaling_convergence/experiment_notes_public.md`
Figures:
- `results/report_figures/fig04_mpi_weak_parallel_execution.svg`
- `results/report_figures/fig05_mpi_weak_computation_time.svg`
- `results/report_figures/fig06_mpi_weak_observed_ratio.svg`

### MPI weak scaling: fixed 88 iterations

A separate experiment uses synthetic data with 10,000 observations per
rank, 8 features, K=8, and exactly 88 iterations. It evaluates six rank
counts, with one warm-up and five measured runs per configuration.

| MPI ranks | Median parallel execution (s) |
|---:|---:|
| 1 | 0.051231 |
| 2 | 0.051385 |
| 4 | 0.053150 |
| 8 | 0.053824 |
| 16 | 0.059214 |
| 32 | 0.061606 |

The 32-rank median is approximately 20.25% higher than the one-rank
median. Only the one-rank configuration converged within 88 iterations.
The other configurations completed 88 iterations but did not meet the
convergence threshold within that limit.

Data: `results/mpi_weak_scaling_fixed88/`
Figure: `results/report_figures/fig10_mpi_fixed_weak_scaling.svg`

### Hybrid MPI and OpenMP

The controlled hybrid experiment uses the full Covertype dataset
(581,012 observations, 55 features, K=7) on one compute node.
Each of seven configurations used one warm-up and three measured runs.
All measured runs converged in 88 iterations with consistent SSE.

| MPI ranks | OpenMP threads per rank | Median runtime (s) |
|---:|---:|---:|
| 1 | 1 | 10.8267 |
| 1 | 2 | 6.0409 |
| 1 | 4 | 3.2672 |
| 2 | 1 | 5.4542 |
| 2 | 2 | 3.0923 |
| 4 | 1 | 2.7423 |
| 4 | 2 | 1.5719 |

The 4-rank, 2-thread configuration achieved approximately 6.89x
speedup relative to the 1-rank, 1-thread baseline (approximately
86.1% efficiency over eight CPU workers). These are single-node
measurements, not multi-node hybrid scaling results.

Data: `results/hybrid_covertype/`
Public CSV: `hybrid_covertype_runs_public.csv`; the internal hostname column is excluded.
Figure: `results/report_figures/fig07_hybrid_configurations.svg`

### CPU and GPU comparison

The CPU/GPU experiment uses the full Covertype dataset, K=7,
one MPI rank, and two OpenMP threads. Five measured CPU/GPU pairs
were collected after warm-up.

| Measurement | CPU | GPU |
|---|---:|---:|
| Median runtime (s) | 9.392093 | 4.626974 |
| Mean runtime (s) | 9.451751 | 4.646658 |
| Sample standard deviation (s) | 0.232041 | 0.039375 |

The median GPU speedup is approximately 2.03x, a runtime reduction
of approximately 50.84%. All measured runs converged in 88 iterations
and reported consistent SSE.

The GPU was an NVIDIA A40 using CUDA 12.9 and compute capability 8.6.
The implementation uses OpenMP target offloading. These are end-to-end
application runtimes, not GPU-kernel-only timings; CPU work and data
transfer overheads are included where applicable.

Data: `results/hybrid_gpu/BENCHMARK_SUMMARY_public.txt`
Figures:
- `results/report_figures/fig08_cpu_gpu_runtime.svg`
- `results/report_figures/fig09_cpu_gpu_speedup.svg`

## Data preparation

The original Forest CoverType dataset is not included in this repository.

The scripts under `scripts/` support dataset preparation and inspection:

- `prepare_covertype.py`
- `inspect_covertype.py`

The preparation script expects the compressed raw dataset at
`data/raw/covtype.data.gz` and writes processed files under
`data/processed/`.

It creates 100k, 500k, and full dataset variants using a fixed random
seed. It does not generate every subset size used in the serial
problem-size experiment.

The raw dataset must be obtained separately from its original provider.
Raw and processed datasets should not be committed to this repository.

## Software and hardware environment

Experiments were performed on the NUS Vanda HPC environment. The
recorded software environments vary by experiment.

| Experiment | Recorded toolchain / hardware |
|---|---|
| Serial problem-size benchmark | GCC 13.1.0 |
| Controlled MPI hybrid benchmark | GCC 13.3.0, OpenMPI 5.0.3 |
| CPU/GPU benchmark | NVIDIA HPC SDK 26.1, CUDA 12.9 |
| GPU device | NVIDIA A40, compute capability 8.6 |

Reproduction should match the compiler and module environment recorded
for the relevant experiment. Results from different toolchains or
hardware configurations should not be compared without accounting
for those differences.

## Reproducibility and provenance

The CSV files preserve measured run-level results where available.
Experiment notes and provenance files document configurations,
timing definitions, and validation information.

Important qualifications:

1. The exact source snapshot used to build the primary MPI
   strong-scaling executable has not been established. The MPI source
   included here is the available benchmark implementation, but it
   is not claimed to be a byte-for-byte match to that executable.

2. The fixed-88-iteration MPI weak-scaling source is the source
   snapshot associated with that experiment.

3. The hybrid controlled source is the source snapshot associated
   with the full-dataset hybrid experiment.

4. Convergence-driven weak-scaling results have different iteration
   counts across rank configurations.

5. In the fixed-88 weak-scaling experiment, only the one-rank
   configuration converged within the 88-iteration limit. Other
   configurations completed the prescribed iterations without
   meeting the convergence threshold.

6. The historical MPI8/MPI256 comparison is supplementary and is
   based on one measured observation per configuration.

The benchmark CSVs and notes should be interpreted together.
Runtime values alone do not establish identical source code,
compiler settings, hardware, or timing boundaries across experiments.

## Figures

Ten SVG figures are included under `results/report_figures/`:

1. `fig01_serial_scaling.svg`
2. `fig02_mpi_strong_runtime.svg`
3. `fig03_mpi_strong_speedup.svg`
4. `fig04_mpi_weak_parallel_execution.svg`
5. `fig05_mpi_weak_computation_time.svg`
6. `fig06_mpi_weak_observed_ratio.svg`
7. `fig07_hybrid_configurations.svg`
8. `fig08_cpu_gpu_runtime.svg`
9. `fig09_cpu_gpu_speedup.svg`
10. `fig10_mpi_fixed_weak_scaling.svg`

## Scope and limitations

This is a curated project archive, not a complete snapshot of an HPC
home or scratch directory.

It intentionally excludes:

- The separately submitted project report.
- Raw and processed datasets.
- Compiled executables and object files.
- PBS scripts, scheduler logs, node allocation files, and job identifiers.
- Temporary files and unrelated intermediate experiments.

Not every experiment has an identical source-to-executable provenance
record. The qualifications above should be retained when interpreting
or reproducing the results.
