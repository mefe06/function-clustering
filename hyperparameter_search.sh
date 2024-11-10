#!/bin/bash

CONDA_ENV="efe_fc"

# Activate the Conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Define the grid of parameters
data_type="m4"
data_path="Extracted_M4"
log_file="hourly_m4_lgbm_hyperparameters_100_series.txt"
model_type="lgbm"
smart_initialization="True"
stochastic_assignments="True"
max_iterations_list=(10 20 40)
seed_list=(2)
model_nb_list=(3 4 5)
test_window_list=(200)
max_depth_list=(3 4 5)
n_iters_list=(100 150 200)
learning_rate_list=(0.1 0.15 0.2)
temperature_list=(1.25)
predict_future_hours=(1)
m4_type="Hourly"
for n_iters in "${n_iters_list[@]}"; do 
    for learning_rate in "${learning_rate_list[@]}"; do 
        for max_depth in "${max_depth_list[@]}"; do 
            for test_window in "${test_window_list[@]}"; do
                for seed in "${seed_list[@]}"; do
                    for max_iterations in "${max_iterations_list[@]}"; do
                        for model_nb in "${model_nb_list[@]}"; do 
                            for temperature in "${temperature_list[@]}"; do
                                for predict_future_hour in "${predict_future_hours[@]}"; do
                                        cmd="python3 main.py --m4_type $m4_type --n_iter $n_iters --learning_rate $learning_rate --weights \"$weights\" --temperature $temperature --seed $seed --test_window $test_window --model_nb $model_nb --predict_future_hours $predict_future_hour --model_type $model_type --log_file $log_file --data_type $data_type --data_path $data_path --max_iterations $max_iterations --stochastic_assignments $stochastic_assignments --smart_initialization $smart_initialization --max_depth $max_depth --num_windows $window"
                                        echo "Running: $cmd"
                                        $cmd
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
