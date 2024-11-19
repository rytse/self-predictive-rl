import subprocess
import time
import csv
from datetime import datetime

def run_experiment(env_name, aux, num_steps, wass_steps=None):
    # Build command based on whether wass_steps is needed
    if wass_steps is not None:
        cmd = f"python main.py --env_name {env_name} --aux {aux} --wass_critic_train_steps {wass_steps} --num_steps {num_steps}"
    else:
        cmd = f"python main.py --env_name {env_name} --aux {aux} --num_steps {num_steps}"
    
    start_time = time.time()
    process = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    end_time = time.time()
    
    if process.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"Error output: {process.stderr.decode()}")
        return None
    
    return end_time - start_time

def main():
    # Environment name
    env_name = "MiniGrid-LavaCrossingS9N3-v0"
    
    # Experiment parameters
    # Simple aux types (no wass_steps)
    simple_aux_types = ['ZP', 'bisim']
    # Aux types that use wass_steps
    critic_aux_types = ['bisim_critic', 'zp_critic']
    wass_steps_values = [1, 5, 10]
    num_steps_values = [100000, 200000, 300000]
    
    # Prepare CSV file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'minigrid_profiling_results_{timestamp}.csv'
    
    # CSV header
    fieldnames = ['env_name', 'aux', 'wass_critic_train_steps', 'num_steps', 'time_seconds']
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Run experiments for simple aux types (no wass_steps)
        for aux in simple_aux_types:
            for num_steps in num_steps_values:
                print(f"\nRunning experiment: aux={aux}, num_steps={num_steps}")
                
                execution_time = run_experiment(env_name, aux, num_steps)
                
                if execution_time is not None:
                    result = {
                        'env_name': env_name,
                        'aux': aux,
                        'wass_critic_train_steps': 'N/A',
                        'num_steps': num_steps,
                        'time_seconds': round(execution_time, 2)
                    }
                    
                    writer.writerow(result)
                    print(f"Completed in {round(execution_time, 2)} seconds")
                    csvfile.flush()
        
        # Run experiments for critic aux types (with wass_steps)
        for aux in critic_aux_types:
            for wass_steps in wass_steps_values:
                for num_steps in num_steps_values:
                    print(f"\nRunning experiment: aux={aux}, wass_steps={wass_steps}, num_steps={num_steps}")
                    
                    execution_time = run_experiment(env_name, aux, num_steps, wass_steps)
                    
                    if execution_time is not None:
                        result = {
                            'env_name': env_name,
                            'aux': aux,
                            'wass_critic_train_steps': wass_steps,
                            'num_steps': num_steps,
                            'time_seconds': round(execution_time, 2)
                        }
                        
                        writer.writerow(result)
                        print(f"Completed in {round(execution_time, 2)} seconds")
                        csvfile.flush()

if __name__ == "__main__":
    main()
