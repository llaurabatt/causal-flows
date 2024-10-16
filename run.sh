#!/bin/bash

# List of configuration files to use
# CONFIG_FILES=(
#     "causal_nf/configs/FF_configs_ate1_model3_0.yaml"
#     # "causal_nf/configs/causal_nf3.yaml"
#     # Add more config files here
# )

for ATE in 1 5; do
    for MODEL in 3; do
        for VERSION in 2 3 4; do
            CONFIG_FILES+=("causal_nf/configs/add_noise_False/FF_configs_ate${ATE}_model${MODEL}_${VERSION}.yaml")
        done
    done
done

# Wandb mode and project name
WANDB_MODE="disabled"
PROJECT_NAME="CAUSAL_NF"

# Iterate over the list of config files and run the Python script with each
for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
    echo "Running script with config file: $CONFIG_FILE"
    python main.py --config_file "$CONFIG_FILE" --wandb_mode "$WANDB_MODE" --project "$PROJECT_NAME"
    
    # Check if the last command was successful
    if [ $? -ne 0 ]; then
        echo "Script failed for config file: $CONFIG_FILE"
        exit 1
    fi
done

echo "All scripts executed successfully!"