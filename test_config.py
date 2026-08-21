from src.config import load_config

config = load_config("configs/glucose.yaml")

print(config["experiment"]["id"])
print(config["covariates"]["adjustment"])
print(config["policy"]["budget_fraction"])
