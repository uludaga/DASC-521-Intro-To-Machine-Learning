import numpy as np
import pandas as pd

X = np.genfromtxt("hw01_data_points.csv", delimiter=",", dtype=str)
y = np.genfromtxt("hw01_class_labels.csv", delimiter=",", dtype=int)


# STEP 3
# first 50000 data points should be included to train
# remaining 44727 data points should be included to test
# should return X_train, y_train, X_test, and y_test
def train_test_split(X, y):
    # your implementation starts below
    n_train = 50000
    # n_test = 44727
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    # your implementation ends above
    return (X_train, y_train, X_test, y_test)


X_train, y_train, X_test, y_test = train_test_split(X, y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# STEP 4
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below
    # # of classes
    K = np.max(y)

    class_priors = [np.mean([y == (c + 1)]) for c in range(K)]
    # your implementation ends above
    return (class_priors)


class_priors = estimate_prior_probabilities(y_train)
print(class_priors)


# STEP 5
# assuming that there are K classes and D features
# should return four numpy arrays with shape (K, D)
def estimate_nucleotide_probabilities(X, y):
    # your implementation starts below

    K = np.unique(y)
    nuc_tags = np.array(["A", "C", "G", "T"])

    nuc_filter = (X[:, :, np.newaxis] == nuc_tags).astype(int)

    # Calculate model parameter array
    mod_param = np.array([np.mean(nuc_filter[y == c], axis=0) for c in K])

    # the first item A is zero, the last item T is 3
    pAcd = mod_param[:, :, 0]
    pCcd = mod_param[:, :, 1]
    pGcd = mod_param[:, :, 2]
    pTcd = mod_param[:, :, 3]
    # your implementation ends above
    return(pAcd, pCcd, pGcd, pTcd)


pAcd, pCcd, pGcd, pTcd = estimate_nucleotide_probabilities(X_train, y_train)
print(pAcd)
print(pCcd)
print(pGcd)
print(pTcd)


# STEP 6
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, pAcd, pCcd, pGcd, pTcd, class_priors):
    # your implementation starts below
    N, D = X.shape[0], X.shape[1]
    K = len(class_priors)

    # Log of class_priors
    log_class_priors = np.log(class_priors)

    # Store nucleotide indices
    nucl_indices = np.zeros((N, D, 4), dtype=int)

    # {A: 0, C: 1, G: 2, T: 3}
    for i, base in enumerate(['A', 'C', 'G', 'T']):
        nucl_indices[:, :, i] = (X == base).astype(int)

    # Loglikelihood
    log_likelihood = np.zeros((N, K, D))
    for k in range(K):
        log_likelihood[:, k, :] = (np.log(pAcd[k]) * nucl_indices[:, :, 0] +
                                   np.log(pCcd[k]) * nucl_indices[:, :, 1] +
                                   np.log(pGcd[k]) * nucl_indices[:, :, 2] +
                                   np.log(pTcd[k]) * nucl_indices[:, :, 3])
    score_values = log_likelihood.sum(axis=2) + log_class_priors
    # your implementation ends above
    return (score_values)


scores_train = calculate_score_values(X_train, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_train)

scores_test = calculate_score_values(X_test, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_test)


# STEP 7
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    K = np.max(y_truth)
    # We need a KxK numpy zero-array to obtain the confusion matrix size of KxK
    confusion_matrix = np.zeros((K, K), dtype=int)
    # Get predictions
    y_pred = np.argmax(scores, axis=1)
    for true_class in range(1, K + 1):
        for pred_class in range(K):
            confusion_matrix[true_class - 1, pred_class] = np.sum((y_truth == true_class) & (y_pred == pred_class))
    confusion_matrix[0, 1], confusion_matrix[1, 0] = confusion_matrix[1, 0], confusion_matrix[0, 1]
    # your implementation ends above
    return (confusion_matrix)


confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)

confusion_test = calculate_confusion_matrix(y_test, scores_test)
print(confusion_test)