# MassDeviation CUDA

Two-component Lattice Boltzmann Method (LBM) simulation of a static droplet using the D2Q9 scheme with a recoloring/segregation operator and interfacial tension source term.

Two implementations are provided:

| File | Back-end | 
|---|---|
| `main.c` | CPU (C99) | 
| `main.cu` | GPU (CUDA) |

---

## Dependencies

### CPU (`main.c`)
- GCC ≥ 9 with C99 support

### GPU (`main.cu`)
- NVIDIA CUDA Toolkit ≥ 11 (`nvcc`)
- A CUDA-capable GPU
- OpenMP (for the `-Xcompiler -fopenmp` flag)

### Benchmark (`benchmark.py`)
- Python ≥ 3.10
- pandas, matplotlib 

---

## Key parameters

These are compile-time macros defined at the top of each source file. Override them with `-D` flags at compile time.

| Macro | Default | Description |
|---|---|---|
| `nL` | 100 | Grid width (number of lattice nodes in x) |
| `nH` | 100 | Grid height (number of lattice nodes in y) |
| `nt` | 1 000 000 | Maximum number of time steps |
| `R` | 25 | Initial droplet radius (lattice units) |

---

## Compiling and running

### CPU — `main.c`

**Default build** (uses the values hard-coded in the file):
```bash
gcc -O3 -fopenmp -lm main.c -o main_cpu
./main_cpu
```

**Custom parameters** via `-D` flags:
```bash
gcc -O3 -fopenmp -lm -DnL=200 -DnH=200 -Dnt=500000 -DR=50 main.c -o main_cpu
./main_cpu
```

Output files written after the run:
- `ux.txt`, `uy.txt` — velocity field
- `rho_p.txt`, `rho_q.txt` — component densities
- `timeCPU.txt` — wall-clock execution time (appended)
- `erro_nH*_nL*_R*_...txt` — convergence error history

---

### GPU — `main.cu`

**Default build:**
```bash
nvcc -O3 -Xcompiler -fopenmp -lm main.cu -o main_cuda
./main_cuda
```

**Custom parameters:**
```bash
nvcc -O3 -Xcompiler -fopenmp -lm -DnL=200 -DnH=200 -Dnt=500000 -DR=50 main.cu -o main_cuda
./main_cuda
```

Output files are identical to the CPU version, except timing is written to `timeGPU.txt`.

---

## Benchmark script — `benchmark.py`

`benchmark.py` sweeps a parameter grid, compiles a fresh binary for each combination, measures wall-clock execution time, and saves results to `benchmark_results.csv`.

### Install dependencies
```bash
pip install pandas,matplotlib
```

### Edit the parameter grid

Open `benchmark.py` and adjust the lists near the top:
```python
NL_VALUES  = [100, 200]       # grid widths to sweep
NH_VALUES  = [100, 200]       # grid heights to sweep
NT_VALUES  = [1_000_000]      # time step counts to sweep
```

`R` is derived automatically as `nL // 4` for each combination. Set `R_OVERRIDE` to a fixed list if needed.

### Run

```bash
# Benchmark the CUDA binary only (default)
python benchmark.py

# Benchmark the CPU binary only
python benchmark.py --target cpu

# Benchmark both back-ends and compare
python benchmark.py --target both
```

Compiled binaries are placed in `benchmark/build/`. Each binary is uniquely named so runs do not overwrite each other.

### Output

- Live progress is printed to the console, including MLUP/s (mega lattice-updates per second) for each run.
- `benchmark_results.csv` — one row per run with columns: `target`, `nL`, `nH`, `nt`, `R`, `grid_size`, `compile_ok`, `execution_time`, `error`.

Example output:
```
[1/36] nL10_nH10_nt1000_R2_cuda
         Compiling ...  OK  →  running (output → nL10_nH10_nt1000_R2_cuda/) ...  0.50s  (0.2 MLUP/s)
[2/36] nL10_nH10_nt1000_R2_cpu
         Compiling ...  OK  →  running (output → nL10_nH10_nt1000_R2_cpu/) ...  0.03s  (3.6 MLUP/s)
[3/36] nL10_nH10_nt1000000_R2_cuda
         Compiling ...  OK  →  running (output → nL10_nH10_nt1000000_R2_cuda/) ...  1.02s  (98.0 MLUP/s)
[4/36] nL10_nH10_nt1000000_R2_cpu
         Compiling ...  OK  →  running (output → nL10_nH10_nt1000000_R2_cpu/) ...  0.13s  (760.3 MLUP/s)
...
Saved to: benchmark_results.csv
```
