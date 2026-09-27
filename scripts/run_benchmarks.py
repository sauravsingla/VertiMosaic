from vertimosaic.experiments import run_cpu_benchmarks

if __name__ == "__main__":
    print(run_cpu_benchmarks([10_000, 30_000, 50_000, 100_000, 250_000], seed=42))
