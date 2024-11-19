import subprocess
import time
import csv
from datetime import datetime

def run_experiment(aux, wass_steps, train_steps):
    cmd = f"python train.py id=HalfCheetah-v2 aux={aux} wass_critic_train_steps={wass_steps} save_dir=logs_benchmark num_train_steps={train_steps}"
    
    start_time = time.time()
    process = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    end_time = time.time()
    
    if process.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"Error output: {process.stderr.decode()}")
        return None
    
    return end_time - start_time

def main():
    # Experiment parameters
    aux_types = ['bisim_critic', 'zp_critic']
    wass_steps_values = [1, 5, 10]
    train_steps_values = [10000, 20000, 30000]
    
    # Prepare CSV file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'profiling_results_{timestamp}.csv'
    
    # CSV header
    fieldnames = ['aux', 'wass_critic_train_steps', 'num_train_steps', 'time_seconds']
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Run experiments
        for aux in aux_types:
            for wass_steps in wass_steps_values:
                for train_steps in train_steps_values:
                    print(f"\nRunning experiment: aux={aux}, wass_steps={wass_steps}, train_steps={train_steps}")
                    
                    execution_time = run_experiment(aux, wass_steps, train_steps)
                    
                    if execution_time is not None:
                        result = {
                            'aux': aux,
                            'wass_critic_train_steps': wass_steps,
                            'num_train_steps': train_steps,
                            'time_seconds': round(execution_time, 2)
                        }
                        
                        writer.writerow(result)
                        print(f"Completed in {round(execution_time, 2)} seconds")
                        
                        # Flush the CSV file to save results immediately
                        csvfile.flush()

if __name__ == "__main__":
    main()
