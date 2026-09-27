from vertimosaic.experiments import run_synthetic_experiment

if __name__ == "__main__":
    print(run_synthetic_experiment(rows=2000, seed=42, model_name="logistic"))
