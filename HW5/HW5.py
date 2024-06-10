import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.spatial.distance as dt
import scipy.stats as stats

group_means = np.array([[-6.0, -1.0],
                        [-3.0, +2.0],
                        [+3.0, +2.0],
                        [+6.0, -1.0]])

group_covariances = np.array([[[+0.4, +0.0],
                               [+0.0, +4.0]],
                              [[+2.4, -2.0],
                               [-2.0, +2.4]],
                              [[+2.4, +2.0],
                               [+2.0, +2.4]],
                              [[+0.4, +0.0],
                               [+0.0, +4.0]]])

# read data into memory
data_set = np.genfromtxt("hw05_data_set.csv", delimiter = ",")

# get X values
X = data_set[:, [0, 1]]

# set number of clusters
K = 4

# STEP 2
# should return initial parameter estimates
# as described in the homework description
def initialize_parameters(X, K):
    # your implementation starts below
    centroids = np.genfromtxt("hw05_initial_centroids.csv", delimiter=",")
    
    means = np.zeros((K, X.shape[1]))
    covariances = np.zeros((K, X.shape[1], X.shape[1]))
    priors = np.zeros(K)

    centroid_distances = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in centroids])
    closest_centroids = np.argmin(centroid_distances, axis=0)

    for k in range(K):
        cluster_points = X[closest_centroids == k]
        means[k] = np.mean(cluster_points)

        covariances[k] = np.cov(cluster_points, rowvar=False)

        priors[k] = len(cluster_points) / len(X)
    # your implementation ends above
    return(means, covariances, priors)

means, covariances, priors = initialize_parameters(X, K)

# STEP 3
# should return final parameter estimates of
# EM clustering algorithm
def em_clustering_algorithm(X, K, means, covariances, priors):
    # your implementation starts below
    N = X.shape[0]
    probs = np.zeros((N, K))

    for iteration in range(100):
        for k in range(K):
            rv = stats.multivariate_normal(means[k], covariances[k])
            probs[:, k] = priors[k] * rv.pdf(X)
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        prob_sum = probs.sum(axis=0)
        for k in range(K):
            means[k] = (probs[:, k, None] * X).sum(axis=0) / prob_sum[k]
            x_centered = X - means[k]
            covariances[k] = (probs[:, k, None, None] * np.matmul(x_centered[:, :, None], x_centered[:, None, :])).sum(axis=0) / prob_sum[k]
            priors[k] = prob_sum[k] / N

    assignments = np.argmax(probs, axis=1)
    # your implementation ends above
    return(means, covariances, priors, assignments)

means, covariances, priors, assignments = em_clustering_algorithm(X, K, means, covariances, priors)
print(means)
print(priors)

# STEP 4
# should draw EM clustering results as described
# in the homework description
def draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments):
    # your implementation starts below
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])

    x_min, x_max = -8.2, 8.2
    y_min, y_max = -8.2, 8.2
    x_mesh, y_mesh = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    for k in range(K):
        plt.scatter(X[assignments == k, 0], X[assignments == k, 1], s=5, color=cluster_colors[k])

    def plot_org_gaussian(mean, cov, color):
        rv = stats.multivariate_normal(mean, cov)
        Z = rv.pdf(np.c_[x_mesh.ravel(), y_mesh.ravel()])
        Z = Z.reshape(x_mesh.shape)
        plt.contour(x_mesh, y_mesh, Z, levels=[0.01], colors=color, linestyles='dashed', linewidths=1)
        
    def plot_EM_gaussian(mean, cov, color):
        rv = stats.multivariate_normal(mean, cov)
        Z = rv.pdf(np.c_[x_mesh.ravel(), y_mesh.ravel()])
        Z = Z.reshape(x_mesh.shape)
        plt.contour(x_mesh, y_mesh, Z, levels=[0.01], colors=color, linestyles='solid', linewidths=1)

    # Original Gaussian densities
    for k in range(K):
        plot_org_gaussian(group_means[k], group_covariances[k], 'black')

    # Gaussian densities obtained by EM algorithm
    for k in range(K):
        plot_EM_gaussian(means[k], covariances[k], cluster_colors[k])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()
    # your implementation ends above
    
draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments)