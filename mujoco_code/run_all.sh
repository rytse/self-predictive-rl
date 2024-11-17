#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 <envs_file> [options]"
    echo "  <envs_file>: Path to file containing environment names"
    echo "  [options]: Additional options passed directly to main.py along with --env_name"
    echo "Example: $0 my_envs.txt aux=bisim_critic device=cuda:0"
    exit 1
}

# Check if at least one argument is provided
if [ $# -lt 1 ]; then
    usage
fi

# Store the first argument as the environments file and remove it from args
envs_file="$1"
shift

# Check if the specified environments file exists
if [ ! -f "$envs_file" ]; then
    echo "Error: File '$envs_file' not found!"
    exit 1
fi

# Count total number of non-empty and non-commented lines
total_envs=$(grep -v '^#' "$envs_file" | grep -v '^[[:space:]]*$' | wc -l)
current_env=0

echo "Using environments file: $envs_file"
echo "Found $total_envs environments to process"
[ $# -gt 0 ] && echo "Command line arguments: $@"
echo "----------------------------------------"

# Read each line from the environments file and run the command with the corresponding environment
while IFS= read -r env_name; do
    # Skip empty lines and lines starting with #
    [[ -z "$env_name" ]] && continue
    [[ "$env_name" =~ ^#.*$ ]] && continue
    
    ((current_env++))
    remaining=$((total_envs - current_env))
    
    echo "Progress: [$current_env/$total_envs] ($remaining remaining)"
    echo "Running experiment with environment: $env_name"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    
    if [ $# -gt 0 ]; then
        # Run with additional arguments if provided
        python train.py id="$env_name" "$@"
    else
        # Run with just the environment if no additional arguments
        python train.py id="$env_name"
    fi
    
    # Capture the exit status
    status=$?
    if [ $status -ne 0 ]; then
        echo "Warning: Command exited with status $status"
    fi
    
    echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Completed experiment with environment: $env_name"
    echo "----------------------------------------"
    
done < "$envs_file"

echo "All experiments completed!"
echo "Total environments processed: $total_envs"
echo "Final completion time: $(date '+%Y-%m-%d %H:%M:%S')"