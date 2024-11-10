from function_clustering import FunctionClustering
from utils import plot_training_losses_horizontal, plot_ensemble_weights
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor 
from sklearn.neural_network import MLPRegressor
import argparse
from wind_data_handler import wind_data_handler
from m4_data_handler import process_m4_data
from crime_data_handler import crime_data_handler
import matplotlib.pyplot as plt
import numpy as np
import os
import random

def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_type", default="mlp")
    parser.add_argument("--model_nb",  default=3, type=int) #4 for m4-mlp,
    parser.add_argument("--stochastic_assignments", default= True, type=bool)
    parser.add_argument("--smart_initialization", default= True, type=bool)
    parser.add_argument("--log_file", default="paper_tries_final.txt")
    parser.add_argument("--data_type", default="m4", type=str)
    parser.add_argument("--data_path", default="Extracted_M4", type=str)
    parser.add_argument("--max_iterations", default=20,  type=int)
    parser.add_argument("--predict_future_hours", default=24,  type=int)
    parser.add_argument("--seed", default=4,  type=int) 
    parser.add_argument("--test_window", default=200,  type=int)
    parser.add_argument("--max_depth", default=5,  type=int)
    parser.add_argument("--num_leaves", default=31,  type=int)
    parser.add_argument("--temperature", default=1.25,  type=float)
    parser.add_argument("--n_iter", default=100, type=int) 
    parser.add_argument("--learning_rate", default=0.01, type=float) 
    parser.add_argument("--model_path", default="empty", type=str)
    parser.add_argument("--hidden_layer_sizes", default="100", type=str) 
    parser.add_argument("--m4_type", default="Weekly", type=str)
    parser.add_argument("--fc_lr", default=1.5, type=float)
    parser.add_argument("--zeta", default=0.05, type=float)
    parser.add_argument("--batch_size", default=400, type=int)
    args=parser.parse_args()
    return args

def log_to_file(file_path, test_setup, content):
    file = open(f"test_results/{file_path}", "a")
    file.write("-------")
    file.write("\n\n")
    file.write(f"Hidden layers: {test_setup['hidden_layers']}, MLP iters: {test_setup['MLP iters']}, MLP Lr: {test_setup['MLP Lr']}, temperature: {test_setup['temperature']}, seed: {test_setup['seed']}, model_nb: {test_setup['model_nb']}, is_stochastic_assignments: {test_setup['is_stochastic_assignments']}, is_smart_initialization: {test_setup['is_smart_initialization']}, iterations: {test_setup['iterations']}, zeta: {test_setup['zeta']}, batch_size: {test_setup['batch_size']}, fc_lr: {test_setup['fc_lr']}")
    file.write("\n")
    file.write(content)
    file.close()

def main():
    args = parse_args()
    set_seeds(args.seed)
    number_of_models = args.model_nb
    log_file = args.log_file
    hidden_layer_sizes = [int(x.strip()) for x in args.hidden_layer_sizes.strip('"').split(',')]    
    is_stochastic_assignments = args.stochastic_assignments
    is_smart_initialization = args.smart_initialization
    number_of_models = args.model_nb
    data_type = args.data_type
    predict_future_hours = args.predict_future_hours
    path = args.data_path
    max_iterations = args.max_iterations
    test_window = args.test_window
    seed = args.seed
    temperature = args.temperature
    m4_type = args.m4_type
    fc_lr = args.fc_lr
    zeta = args.zeta
    batch_size = args.batch_size
    if args.model_type == "mlp":
        model = MLPRegressor
        model_args = {"hidden_layer_sizes":hidden_layer_sizes, "max_iter":args.n_iter, "learning_rate_init":args.learning_rate, "batch_size":args.batch_size}
    else: 
        model = LGBMRegressor
        model_args = {'max_depth': args.max_depth, 'num_iterations': args.n_iter, 'learning_rate': args.learning_rate}
    val=True
    train_models = True
    test_models = True
    save_models = True
    model_path = args.model_path
    if args.model_path == "empty":
        model_path =f"models/{args.model_type}_{data_type}_{data_type}_{number_of_models}_{seed}_{temperature}_{args.max_depth}_{args.n_iter}_{args.learning_rate}.pkl"
    if data_type.find("wind") != -1:
        data, test_data = wind_data_handler(path=path, predict_future_hours=predict_future_hours)
        is_labels_at_end = False
    elif data_type.find("m4") != -1:
        data, test_data = process_m4_data(folder_path=path, n=100, frequency=m4_type)
        is_labels_at_end = False
    else:
        data, test_data = crime_data_handler(path, aware_split = True)
        is_labels_at_end = False

    fc = FunctionClustering(model=model, data=data, model_kwargs=model_args, loss_fn=mean_squared_error, number_of_models=number_of_models, max_iterations=max_iterations, is_labels_at_end= is_labels_at_end, learning_rate=fc_lr, seed=seed, zeta=zeta) #, n_jobs=15)
    if train_models:
        clusters, models = fc.create_clusters(stochastic_assignments=is_stochastic_assignments, smart_initialization=is_smart_initialization)
        if save_models:
            fc.save_model(model_path)
        plot_training_losses_horizontal(fc.cluster_cost_history)
    else:
        fc.load_model(model_path)
    if test_models: 
        errors = fc.test_loop(test_data, test_window)
        ensemble_weights = fc.ensemble_weights_memory
        plot_ensemble_weights(ensemble_weights)
        test_setup = {"hidden_layers": hidden_layer_sizes, "MLP iters": args.n_iter, "MLP Lr": args.learning_rate,"temperature":temperature, "seed":seed, "model_nb":number_of_models , "is_stochastic_assignments": is_stochastic_assignments, "is_smart_initialization": is_smart_initialization, "iterations": "NA", "zeta": zeta, "batch_size": batch_size, "fc_lr": fc_lr} #, "ensemble_weights": ensemble_weights}
        log_to_file(log_file, test_setup, " ".join([str(error)for error in errors])+ "\n \n"+ f"{sum(errors)/len(errors)}")

if __name__ == "__main__":
    main()