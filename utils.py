import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

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
    plt.savefig(f"final_test_results/modelling_cost/modelling_cost_{current_time}.png")
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

    plt.title("Modelling Cost over Iterations")
    plt.xlabel('Iteration')
    plt.ylabel('Modelling Cost (MSE)')
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
    plt.savefig(f"final_test_results/losses/losses_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")

def plot_training_losses_horizontal(training_losses, title="Modelling Cost over Iterations"):
    # Set style for better readability
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 14})

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.suptitle(title, fontsize=18)

    num_iterations = len(training_losses)
    num_clusters = len(training_losses[0])
    
    # Color palette
    colors = sns.color_palette("husl", num_clusters)

    # Plot training losses
    for i in range(num_clusters):
        costs = [iteration[i] for iteration in training_losses]
        ax.plot(range(num_iterations), costs, label=f'Cluster {i+1}', color=colors[i], linewidth=2.5)

    ax.set_xlabel('Iteration', fontsize=16)
    ax.set_ylabel('Modelling Cost (MSE)', fontsize=16)
    ax.legend(fontsize=12, loc='upper right')
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout and save
    plt.tight_layout()
    filename = f"training_losses_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    plt.savefig(f"final_test_results/losses/{filename}", dpi=300, bbox_inches='tight')
    plt.close()

    return filename

def plot_ensemble_weights(ensemble_weights, title="Ensemble Weights over Batches", file=None):
    # Set the style to a more professional look
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    # Define line styles and markers for the plot
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    num_iterations = len(ensemble_weights)
    num_clusters = len(ensemble_weights[0])

    # Plot ensemble weights for each cluster
    for i in range(num_clusters):
        weights = [iteration[i] for iteration in ensemble_weights]
        plt.plot(range(num_iterations), weights,
                 label=f'Cluster {i+1}',
                 linewidth=2,
                 color=colors[i % len(colors)],
                 linestyle=line_styles[i % len(line_styles)])
        
        # Add markers every 5 points
        plt.plot(range(0, num_iterations, 5), weights[::5],
                 marker=markers[i % len(markers)],
                 linestyle='None',
                 markersize=5)

    plt.title(title, fontsize=20, fontweight='bold')
    plt.xlabel('Batches', fontsize=16,  fontweight='bold')
    plt.ylabel('Ensemble Weight', fontsize=16, fontweight='bold')

    # Increase legend size and adjust position
    plt.legend(fontsize=12, title='Clusters', title_fontsize=14,
               loc='upper left', ncol=1, framealpha=0.8)

    # Adjust tick label size
    plt.tick_params(axis='both', which='major', labelsize=10)

    # Add subtle grid lines
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adjust y-axis limits for better visibility
    plt.ylim(min(min(weights) for weights in ensemble_weights) - 0.005,
             max(max(weights) for weights in ensemble_weights) + 0.005)

    plt.tight_layout()

    # Save the plot
    if file:
        filename = f"ensemble_weights_{file}.png"
    else:
        filename = f"ensemble_weights_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    
    plt.savefig(f"final_test_results/weights/{filename}", dpi=300, bbox_inches='tight')
    plt.close()

    return filename
