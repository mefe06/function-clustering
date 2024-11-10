import numpy as np
import random
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor
import os
import joblib

class FunctionClustering():
    def __init__(self, model, data, validation_data, loss_fn, number_of_models, max_iterations, model_kwargs, is_labels_at_end, init_cluster_point_nb = 500, learning_rate = 0.1, seed=42, temperature=0.05, zeta=0.01):   
        self.learning_rate = learning_rate
        self.model_type = model
        self.models = [] 
        self.data = data
        self.number_of_models = number_of_models
        self.loss = loss_fn
        self.max_iterations = max_iterations
        self.init_cluster_point_nb = init_cluster_point_nb
        self.model_kwargs = model_kwargs
        self.is_labels_at_end = is_labels_at_end
        self.prev_cluster_assignments = None
        self.assignment_change_threshold = zeta*len(data)
        self.performance_history = []
        self.cluster_cost_history = []
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.validation_data = validation_data
        self.validation_losses = []
        self.temperature = temperature
        self.true_labels = None
        self.iterations = 0
        self.total_misclassification_ratio = 0.0

    def set_true_labels(self, true_labels):
        self.true_labels = true_labels

    def evaluate_clustering(self):
        if self.true_labels is None:
            raise ValueError("True labels have not been set. Use set_true_labels() method.")
        
        misclassification_rates = []
        total_misclassified = 0
        total_points = 0

        for i, cluster in enumerate(self.clusters):
            cluster = np.array(cluster)
            cluster_indices = []
            for point in cluster:
                matches = np.where(np.all(self.data == point, axis=1))[0]
                if len(matches) > 0:
                    cluster_indices.append(matches[0])
            
            cluster_true_labels = self.true_labels[cluster_indices]
            
            label_counts = np.bincount(cluster_true_labels)
            correct_label = np.argmax(label_counts)
            
            misclassified = np.sum(cluster_true_labels != correct_label)
            total_misclassified += misclassified
            total_points += len(cluster_true_labels)

            misclassification_rate = misclassified / len(cluster_true_labels) if len(cluster_true_labels) > 0 else 0
            misclassification_rates.append(misclassification_rate)
        
        # Update the total misclassification ratio
        self.total_misclassification_ratio = total_misclassified / total_points if total_points > 0 else 0

        return misclassification_rates
        
    def calculate_cluster_costs(self):
        """Calculate the modeling cost for each cluster."""
        costs = []
        for i, model in enumerate(self.models):
            cluster_data = np.array(self.clusters[i])
            if len(cluster_data) > 0:
                X = cluster_data[:, :-1] 
                y = cluster_data[:, -1]
                predictions = model.predict(X)
                cost = self.loss(y, predictions)
                costs.append(cost)
            else:
                costs.append(np.inf)  # Assign infinite cost to empty clusters
        self.cluster_cost_history.append(costs)

    def calculate_validation_loss(self):
        """Calculate the modeling cost for each cluster."""
        costs = []
        for i, model in enumerate(self.models):
            cluster_data = np.array(self.validation_data)
            if len(cluster_data) > 0:
                X = cluster_data[:, :-1]
                y = cluster_data[:, -1]
                predictions = model.predict(X)
                cost = self.loss(y, predictions)
                costs.append(cost)
            else:
                costs.append(np.inf)  # Assign infinite cost to empty clusters
        #self.cluster_costs = costs
        return costs

    def cross_validate_lightgbm(self, X, y):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3]
        }
        best_score = float('-inf')
        best_params = None

        for n_estimators in param_grid['n_estimators']:
            for max_depth in param_grid['max_depth']:
                for learning_rate in param_grid['learning_rate']:
                    model = LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                          learning_rate=learning_rate, random_state=42)
                    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
                    mean_score = np.mean(scores)
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {'n_estimators': n_estimators, 'max_depth': max_depth, 
                                       'learning_rate': learning_rate}

        best_model = LGBMRegressor(**best_params, random_state=42)
        return best_model.fit(X, y)

    def train_model(self, cluster):
        cluster = np.array(cluster)
        return self.cross_validate_lightgbm(cluster[:, :-1], cluster[:, -1])

    def expectation(self):
        self.models = []    
        for cluster in self.clusters:   
            cluster = np.array(cluster)
            model = self.model_type(**self.model_kwargs)
            self.models.append(model.fit(cluster[:, :-1], cluster[:, -1]))
        self.calculate_cluster_costs()

    def calculate_error_matrix(self, data):
        error_matrix = None
        for model in self.models:
            cur_error = (data[:, -1] -model.predict(data[:, :-1])) 
            if error_matrix is None:
                error_matrix = cur_error
            else:
                error_matrix = np.vstack([error_matrix, cur_error])
        return error_matrix
    
    def maximization(self):
        self.clusters = [[] for _ in self.models]
        error_matrix = np.abs(self.calculate_error_matrix(self.data)) #, is_labels_at_end=True))
        best_model_indices = np.array(np.argmin(error_matrix, axis=0))
        errors = np.min(error_matrix, axis=0)
        self.total_errors.append(np.sum(errors))
        for i, model_index in enumerate(best_model_indices):
            self.clusters[model_index].append(self.data[i,:])

    def stochastic_maximization(self):
        self.clusters = [[] for _ in self.models]
        error_matrix = np.power(self.calculate_error_matrix(self.data), 2) #, is_labels_at_end=True), 2)
        errors = np.min(error_matrix, axis=0)
        self.total_errors.append(np.sum(errors))
        model_indices = [i for i in range(self.number_of_models)]
        for i in range(len(self.data)):
            model_probas = self.softmax(-error_matrix[:,i].astype(float)/self.temperature)
            stochastic_model_index= random.choices(model_indices, weights=model_probas, k=1)[0]
            self.clusters[stochastic_model_index].append(self.data[i,:])

    def initialize_clusters(self):
        self.total_errors = [-np.inf]
        self.converged = False
        np.random.shuffle(self.data)
        self.clusters = np.array_split(self.data, self.number_of_models)
        for cluster in self.clusters:
            print(len(cluster))

    def distance_based_loss(self):
        if not self.models:
            return [1/len(self.data)]*len(self.data) ### return uniform distribution
        else:
            error_matrix = np.power(self.calculate_error_matrix(self.data), 2)
            if len(self.models)==1:
                error_matrix = np.array([error_matrix])
            best_errors = np.min(error_matrix, axis=0)
            total_errors = np.sum(error_matrix, axis=0)
            probs = best_errors.astype(float)/ total_errors.astype(float)
            return probs

    def normalize_distribution(self, distribution):
        ## normalize given weights to make sure its a valid simplex
        total = sum(distribution)
        return [prob/total for prob in distribution] 

    def sample_next_centroid(self, distribution):
        sampled_points = random.choices(self.remainining_indices, weights=self.normalize_distribution(distribution), k= self.init_cluster_point_nb)
        self.remainining_indices = [indice for indice in self.remainining_indices if indice not in sampled_points]
        return sampled_points

    def initialize_clusters_kmeans_plus_plus(self):
        self.models = []
        self.remainining_indices = np.arange(len(self.data))
        self.total_errors = [-np.inf]
        self.converged = False
        self.clusters = [[] for _ in range(self.number_of_models)]
        for _ in range(self.number_of_models):
            distribution = self.distance_based_loss()
            distribution = [distribution[i] for i in self.remainining_indices]
            points = self.sample_next_centroid(distribution)
            model = self.model_type()
            self.models.append(model.fit(self.data[points, :-1], self.data[points, -1]))
        self.maximization()

    def calculate_cluster_changes(self):
        current_assignments = np.array([np.argmin(errors) for errors in zip(*[model.predict(self.data[:, :-1]) for model in self.models])])
        if self.prev_cluster_assignments is not None:
            changes = np.sum(current_assignments != self.prev_cluster_assignments)
        else:   
            changes = self.assignment_change_threshold + 1 
        self.prev_cluster_assignments = current_assignments
        return changes
    
    def check_convergence(self):
        changes = self.calculate_cluster_changes()
        if ((changes < self.assignment_change_threshold)or(self.iterations>self.max_iterations)):
            self.converged = True

    def create_clusters(self, stochastic_assignments = False, smart_initialization= False, synthetic=False):
        if smart_initialization:
            self.initialize_clusters_kmeans_plus_plus()
        else:
            self.initialize_clusters()
        self.iterations = 0
        misclassification_history = []
        while not self.converged:
            self.iterations += 1
            self.expectation()
            if stochastic_assignments: 
                self.stochastic_maximization()
            else:
                self.maximization()
            self.check_convergence()
            if synthetic:
                        # Evaluate clustering quality
                misclassification_rates = self.evaluate_clustering()
                misclassification_history.append(misclassification_rates)
            else:
                self.validation_losses.append(self.calculate_validation_loss())
        self.expectation()
        if synthetic:
            return self.clusters, self.models, misclassification_history

        return self.clusters, self.models

    def initialize_ensemble_weights(self):
        self.ensemble_weights = [1/(self.number_of_models) for _ in range(self.number_of_models)]
        self.ensemble_weights = self.softmax(self.ensemble_weights)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def calculate_analytical_gradients(self):
        error = self.accumulated_data[:,-1] - self.predict(self.accumulated_data[:,:-1])
        g_matrix = np.array(self.calculate_error_matrix(self.accumulated_data))
        softmax_weights = self.softmax(self.ensemble_weights)
        J = np.diag(softmax_weights) - np.outer(softmax_weights, softmax_weights)
        analytical_gradients = (2 / len(self.accumulated_data[:,-1])) * J @ g_matrix @ error
        return analytical_gradients
    
    def predict(self, test_data):
        unweighted_predictions = np.array([model.predict(test_data) for model in self.models])
        weighted_predictions = np.matmul(np.array(self.ensemble_weights).T, unweighted_predictions)
        return weighted_predictions

    def calculate_error(self, test_data, test_labels):
        return self.loss(test_labels, self.predict(test_data))

    def model_update(self):
        gradients = self.calculate_analytical_gradients()
        self.ensemble_weights -= self.learning_rate * gradients
        self.ensemble_weights = np.maximum(self.ensemble_weights, 0)
        self.ensemble_weights /= np.sum(self.ensemble_weights)

    def test_loop(self, test_data, window_length):
        errors = []
        self.ensemble_weights_memory = []
        self.initialize_ensemble_weights()
        for window_index in range(0, len(test_data)-window_length+1, window_length):
            self.accumulated_data = test_data[window_index: window_index+window_length, :]
            error = self.calculate_error(self.accumulated_data[:,:-1], self.accumulated_data[:,-1])
            errors.append(error)
            self.performance_history.append(error) 
            self.ensemble_weights_memory.append(self.ensemble_weights.copy())
            self.iterations += 1
            self.model_update()
        return errors

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        for i, model in enumerate(self.models):
            model_path = os.path.join(path, f"model_{i}")
            joblib.dump(model, model_path)
        print(f"Models and information saved to {path}")

    def load_model(self, path):
        self.models = []
        for i in range(self.number_of_models):
            model_path = os.path.join(path, f"model_{i}")
            if not os.path.exists(model_path):
                raise ValueError(f"Model file {model_path} not found")
            model = joblib.load(model_path)
            self.models.append(model)
        
        print(f"Models and information loaded from {path}")