#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options will be passed directly to main.py along with --env_name from envs.txt"
    echo "Example: $0 --aux bisim_critic --device cuda:0"
    exit 1
}

# Check if envs.txt exists
if [ ! -f "envs.txt" ]; then
    echo "Error: envs.txt file not found!"
    exit 1
}

# Count total number of non-empty and non-commented lines
total_envs=$(grep -v '^#' envs.txt | grep -v '^[[:space:]]*$' | wc -l)
current_env=0

echo "Found $total_envs environments to process"
echo "Command line arguments: $@"
echo "----------------------------------------"

# Read each line from envs.txt and run the command with the corresponding environment
while IFS= read -r env_name; do
    # Skip empty lines and lines starting with #
    [[ -z "$env_name" ]] && continue
    [[ "$env_name" =~ ^#.*$ ]] && continue
    
    ((current_env++))
    remaining=$((total_envs - current_env))
    
    echo "Progress: [$current_env/$total_envs] ($remaining remaining)"
    echo "Running experiment with environment: $env_name"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # Run python command with all passed arguments plus the environment
    python main.py \
        --env_name "$env_name" \
        "$@"
    
    # Capture the exit status
    status=$?
    if [ $status -ne 0 ]; then
        echo "Warning: Command exited with status $status"
    fi
    
    echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Completed experiment with environment: $env_name"
    echo "----------------------------------------"
    
done < "envs.txt"

echo "All experiments completed!"
echo "Total environments processed: $total_envs"
echo "Final completion time: $(date '+%Y-%m-%d %H:%M:%S')"
