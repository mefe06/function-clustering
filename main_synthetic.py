from function_clustering import FunctionClustering
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from synthetic_data import generate_synthetic_data
from lightgbm import LGBMRegressor 
from sklearn.neural_network import MLPRegressor
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

def kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters

def calculate_misclassification_rate(true_labels, predicted_labels):
    return np.mean(true_labels != predicted_labels)

def plot_cluster_costs(cluster_cost_history, title="Cluster Costs Over Time"):
    num_iterations = len(cluster_cost_history)
    num_clusters = len(cluster_cost_history[0])
    
    plt.figure(figsize=(12, 6))
    
    for i in range(num_clusters):
        costs = [iteration[i] for iteration in cluster_cost_history]
        plt.plot(range(num_iterations), costs, label=f'Cluster {i+1}')
    
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    
    # Plot mean cost
    mean_costs = [np.mean(iteration) for iteration in cluster_cost_history]
    plt.plot(range(num_iterations), mean_costs, 'k--', label='Mean Cost', linewidth=2)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.tight_layout()
    plt.savefig(f"test_results/modelling_cost_{current_time}.png")
    plt.close()

def plot_losses(training_losses, validation_losses, title="Training and Validation Losses"):
    plt.figure(figsize=(15, 10))
    num_iterations = len(training_losses)
    num_clusters = len(training_losses[0])
    # Plot training losses
    plt.subplot(2, 1, 1)
    for i in range(num_clusters):
        costs = [iteration[i] for iteration in training_losses]
        plt.plot(range(num_iterations), costs, label=f'Cluster {i+1}')

    plt.title("Training Losses")
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot validation losses
    plt.subplot(2, 1, 2)
    num_iterations = len(validation_losses)
    for i in range(num_clusters):
        costs = [iteration[i] for iteration in validation_losses]
        plt.plot(range(num_iterations), costs, label=f'Cluster {i+1}')
    plt.title("Validation Losses")
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"test_results/losses_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

def plot_cluster_scores(histories):
    # Set the style to a more professional look
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    # Define line styles and markers for the plot
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v']

    # Find the maximum length among all histories
    max_length = max(len(history) for history in histories)

    # Calculate the mean score for each cluster across all histories
    n_clusters = len(histories[0][0])
    mean_scores = [[] for _ in range(n_clusters)]

    for i in range(max_length):
        for cluster in range(n_clusters):
            scores_at_i = [history[i][cluster] for history in histories if i < len(history)]
            mean_scores[cluster].append(np.mean(scores_at_i))

    # Plot mean scores for each cluster
    for cluster in range(n_clusters):
        plt.plot(range(max_length), mean_scores[cluster],
                 label=f'Cluster {cluster+1}',
                 linewidth=2,
                 linestyle=line_styles[cluster % len(line_styles)],
                 marker=markers[cluster % len(markers)],
                 markersize=5)

    plt.title('Average Misclassification Rates over Iterations', fontsize=20, fontweight='bold')
    plt.xlabel('Iteration', fontsize=16, fontweight='bold')
    plt.ylabel('Average Misclassification Rate', fontsize=16, fontweight='bold')

    # Increase legend size and adjust position
    plt.legend(fontsize=14, title='Clusters', title_fontsize=12,
               loc='upper right', ncol=1, framealpha=0.8)

    # Adjust tick label size
    plt.tick_params(axis='both', which='major', labelsize=10)

    # Add subtle grid lines
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adjust y-axis to start from 0
    plt.ylim(bottom=0)

    plt.tight_layout()

    # Save the plot
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"final_test_results/cluster_scores_{current_time}.png", dpi=300, bbox_inches='tight')
    plt.close()

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_type", default="mlp")
    parser.add_argument("--model_nb",  default=5, type=int)
    parser.add_argument("--train_data_nb", default=1000, type=int)
    parser.add_argument("--test_data_nb", default=100, type=int)
    parser.add_argument("--feature_nb", default=1, type=int)
    parser.add_argument("--stochastic_assignments", default= True, type=bool)
    parser.add_argument("--smart_initialization", default= True, type=bool)
    parser.add_argument("--log_file", default="good_seeds.txt")
    parser.add_argument("--data_type", default="crime", type=str)
    parser.add_argument("--data_path", default="step_dict.pkl", type=str)
    parser.add_argument("--weight_learning", default="sgd", type=str)
    parser.add_argument("--max_iterations", default=25,  type=int)
    parser.add_argument("--predict_future_hours", default=24,  type=int)
    parser.add_argument("--seed", default=2,  type=int) ## 2 is also v. good.
    parser.add_argument("--test_window", default=200,  type=int)
    parser.add_argument("--max_depth", default=5,  type=int)
    parser.add_argument("--num_leaves", default=31,  type=int)
    parser.add_argument("--temperature", default=1.5,  type=float) #0.8
    parser.add_argument("--weights", default="0.4,0.25,0.2,0.15", type=str) 
    parser.add_argument("--num_windows", default=1, type=int) 
    parser.add_argument("--number_of_sims", default=2, type=int)
    args=parser.parse_args()
    return args

def log_to_file(file_path, test_setup, content):
    file = open(f"test_results/{file_path}", "a")
    file.write("-------")
    file.write("\n\n")
    file.write(f"model_nb: {test_setup['model_nb']}, weights: {test_setup['weights']}, max_depth: {test_setup['max_depth']}, seed: {test_setup['seed']}, test_windows: {test_setup['test_windows']} ,is_stochastic_assignments: {test_setup['is_stochastic_assignments']}, temperature: {test_setup['temperature']},is_smart_initialization: {test_setup['is_smart_initialization']}, converged in: {test_setup['iterations']} iterations. ") # , weights: {test_setup['ensemble_weights']}")
    file.write("\n")
    file.write(f"Predict future hours: {test_setup['predict_future_hours']}")
    file.write("\n")
    file.write(content)
    file.close()

def main():
    args = parse_args()
    number_of_models = args.model_nb
    if args.model_type == "mlp":
        model = MLPRegressor
        model_args = {"hidden_layer_sizes": (100,), "max_iter": 100, "learning_rate_init": 0.01}#{"hidden_layer_sizes":(40, 20)}
    else: 
        model = LGBMRegressor
        num_leaves = args.num_leaves
        model_args = {'num_leaves': num_leaves} #{ 'max_depth': [args.max_depth]}

    log_file = args.log_file
    train_data_nb = args.train_data_nb
    test_data_nb = args.test_data_nb
    feature_nb = args.feature_nb
    weights = [float(x.strip()) for x in args.weights.strip('"').split(',')]
    is_stochastic_assignments = args.stochastic_assignments
    is_smart_initialization = args.smart_initialization
    number_of_models = args.model_nb
    data_type = args.data_type
    predict_future_hours = args.predict_future_hours
    path = args.data_path
    weight_learning = args.weight_learning
    max_iterations = args.max_iterations
    test_window = args.test_window
    seed = args.seed
    temperature = args.temperature
    num_windows = args.num_windows
    number_of_sims = args.number_of_sims
    val=True
    train_data_nb = 5000
    feature_nb = 3 
    fc_misclassification_histories = []
    kmeans_misclassification_histories = []
    last= 0
    number_of_data_models = 3
    misclassification_histories = []
    random_states_1 = [i for i in range(50)]
    seed = 2
    for random_state in random_states_1:
        train_data, test_data, train_labels, test_labels, generating_coeffs = generate_synthetic_data(number_of_data_models, train_data_nb, num_features = feature_nb, random_state=random_state)
        is_labels_at_end=False
        fc = FunctionClustering(model=model, data=train_data, validation_data = None, model_kwargs=model_args, loss_fn=mean_squared_error, number_of_models=number_of_models, max_iterations=max_iterations, is_labels_at_end=is_labels_at_end, learning_rate=0.05, seed=seed, temperature=temperature)
        fc.set_true_labels(train_labels)
        clusters, models, misclassification_history = fc.create_clusters(stochastic_assignments=is_stochastic_assignments, smart_initialization=is_smart_initialization, synthetic=True)
        misclassification_histories.append(misclassification_history)
        fc_misclassification_histories.append(fc.total_misclassification_ratio)
        print(fc.total_misclassification_ratio)
        kmeans_clusters = kmeans_clustering(train_data, n_clusters=number_of_models)
        kmeans_misclassification = calculate_misclassification_rate(train_labels, kmeans_clusters)
        kmeans_misclassification_histories.append(kmeans_misclassification)
        print(f'KMeans misclassification rate: {kmeans_misclassification}')

    file = open(f"final_test_results/synth/results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", "a")
    file.write(f"{last}")
    file.write(f"FC misclassification rates: {fc_misclassification_histories}; KMeans misclassification rates: {kmeans_misclassification_histories}, seed: {seed}, temperature: {temperature}, number_of_sims: {number_of_sims}")
    file.write(f"FC misclassification rate: {sum(fc_misclassification_histories)/len(fc_misclassification_histories)}; KMeans misclassification rate: {sum(kmeans_misclassification_histories)/len(kmeans_misclassification_histories)}, seed: {seed}, temperature: {temperature}, number_of_sims: {number_of_sims}")
    file.close()

if __name__ == "__main__":
    main()