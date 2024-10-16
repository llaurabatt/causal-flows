#!/bin/bash

# List of configuration files to use
CONFIG_FILES=(
    # "causal_nf/configs/FF_configs_ate1_model1_0.yaml"
    "causal_nf/configs/fake2.yaml"
    # "causal_nf/configs/causal_nf3.yaml"
    # Add more config files here
)

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