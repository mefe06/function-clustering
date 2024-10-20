import numpy as np
from sklearn.model_selection import train_test_split

def generate_synthetic_data(num_sources, num_samples, num_features=5, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    X = np.random.randn(num_samples, num_features)
    y = np.zeros(num_samples)
    true_labels = np.zeros(num_samples, dtype=int)
    samples_per_source = num_samples // num_sources
    all_coeffs = []
    for i in range(num_sources):
        start_idx = i * samples_per_source
        end_idx = (i + 1) * samples_per_source if i < num_sources - 1 else num_samples
        func_choice = "linear" #np.random.choice(['linear', 'polynomial'])
        
        if func_choice == 'linear':
            coeffs = np.random.uniform(-10, 10, num_features)
            coeffs[np.random.randint(num_features)] = np.random.choice([-10, 10]) * np.random.uniform(0.8, 1.2)
            y[start_idx:end_idx] = np.dot(X[start_idx:end_idx], coeffs)
            print(coeffs)
            all_coeffs.append(coeffs)
            intercept = np.random.uniform(-50, 50)
            y[start_idx:end_idx] += intercept
            
        else:  # polynomial
            degree = np.random.randint(2, 4)  # Random degree between 2 and 3
            coeffs = np.random.uniform(-5, 5, degree + 1)
            coeffs[0] = np.random.choice([-5, 5]) * np.random.uniform(0.8, 1.2)
            feature_weights = np.random.uniform(-2, 2, num_features)
            feature_combination = np.dot(X[start_idx:end_idx], feature_weights)
            y[start_idx:end_idx] = np.polyval(coeffs, feature_combination)
        scale_factor = np.random.uniform(0.5, 2.0)
        y[start_idx:end_idx] *= scale_factor
        noise_level = np.random.uniform(0.05, 0.15)
        y[start_idx:end_idx] += np.random.normal(0, noise_level * np.std(y[start_idx:end_idx]), end_idx - start_idx)
        
        true_labels[start_idx:end_idx] = i
    data = np.column_stack((X, y))
    shuffled_indices = np.random.permutation(num_samples)
    data = data[shuffled_indices]
    true_labels = true_labels[shuffled_indices]
    train_data, test_data, train_labels, test_labels = train_test_split(data, true_labels, test_size=test_size, random_state=random_state)

    return train_data, test_data, train_labels, test_labels, np.array(all_coeffs)