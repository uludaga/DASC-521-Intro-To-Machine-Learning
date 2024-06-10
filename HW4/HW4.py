import matplotlib.pyplot as plt
import numpy as np


# read data into memory
data_set_train = np.genfromtxt("hw04_data_set_train.csv", delimiter = ",", skip_header = 1)
data_set_test = np.genfromtxt("hw04_data_set_test.csv", delimiter = ",", skip_header = 1)

# get x and y values
X_train = data_set_train[:, 0:1]
y_train = data_set_train[:, 1]
X_test = data_set_test[:, 0:1]
y_test = data_set_test[:, 1]

# set drawing parameters
minimum_value = 1.5
maximum_value = 5.1
step_size = 0.001
X_interval = np.arange(start = minimum_value, stop = maximum_value + step_size, step = step_size)
X_interval = X_interval.reshape(len(X_interval), 1)

def plot_figure(X_train, y_train, X_test, y_test, X_interval, y_interval_hat):
    fig = plt.figure(figsize = (8, 4))
    plt.plot(X_train[:, 0], y_train, "b.", markersize = 10)
    plt.plot(X_test[:, 0], y_test, "r.", markersize = 10)
    plt.plot(X_interval[:, 0], y_interval_hat, "k-")
    plt.xlabel("Eruption time (min)")
    plt.ylabel("Waiting time to next eruption (min)")
    plt.legend(["training", "test"])
    plt.show()
    return(fig)

# STEP 2
# should return necessary data structures for trained tree
def decision_tree_regression_train(X_train, y_train, P):
    # create necessary data structures
    node_indices = {}
    is_terminal = {}
    need_split = {}

    node_features = {}
    node_splits = {}
    node_means = {}
    # your implementation starts below
    N = X_train.shape[0]
    D = X_train.shape[1]

    node_indices[1] = np.arange(N)
    need_split[1] = True
    is_terminal[1] = False

    while True:
        split_nodes = [key for key, value in need_split.items() if value]
        if len(split_nodes) == 0:
            break
        for split_node in split_nodes:
            data_indices = node_indices[split_node]
            need_split[split_node] = False
            node_means[split_node] = np.mean(y_train[data_indices])
            if len(data_indices) <= P:
                is_terminal[split_node] = True
                continue
            is_terminal[split_node] = False
            best_split_score = float('inf')
            best_split_position = None
            for d in range(D):
                unique_values = np.sort(np.unique(X_train[data_indices, d]))
                split_positions = (unique_values[1:] + unique_values[:-1]) / 2

                for split_position in split_positions:
                    left_indices = data_indices[X_train[data_indices, d] > split_position]
                    right_indices = data_indices[X_train[data_indices, d] <= split_position]

                    total_error = 0
                    if len(left_indices) > 0:
                        total_error += np.sum((y_train[left_indices] - np.mean(y_train[left_indices]))**2)
                    if len(right_indices) > 0:
                        total_error += np.sum((y_train[right_indices] - np.mean(y_train[right_indices]))**2)

                    split_score = total_error / len(data_indices)

                    if split_score < best_split_score:
                        best_split_score = split_score
                        best_split_position = split_position
                        best_split_d = d
            node_features[split_node] = best_split_d
            node_splits[split_node] = best_split_position

            left_indices = data_indices[X_train[data_indices, best_split_d] > best_split_position]
            node_indices[2 * split_node] = left_indices
            is_terminal[2 * split_node] = False
            need_split[2 * split_node] = True

            right_indices = data_indices[X_train[data_indices, best_split_d] <= best_split_position]
            node_indices[2 * split_node + 1] = right_indices
            is_terminal[2 * split_node + 1] = False
            need_split[2 * split_node + 1] = True

    # your implementation ends above
    return(is_terminal, node_features, node_splits, node_means)

# STEP 3
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def decision_tree_regression_test(X_query, is_terminal, node_features, node_splits, node_means):
    # your implementation starts below
    N_query = X_query.shape[0]
    y_hat = np.zeros(N_query)
    def traverse_tree(node, query_point):
        if is_terminal[node]:
            return node_means[node]
        split_feature = node_features[node]
        split_value = node_splits[node]
        if query_point[split_feature] > split_value:
            return traverse_tree(2 * node, query_point)
        else:
            return traverse_tree(2 * node + 1, query_point)
    for i in range(N_query):
        y_hat[i] = traverse_tree(1, X_query[i, :])
    return(y_hat)

# STEP 4
# assuming that there are T terminal node
# should print T rule sets as described
def extract_rule_sets(is_terminal, node_features, node_splits, node_means):
    rule_sets = []
    
    terminal_nodes = [key for key, value in is_terminal.items() if value == True]
    
    for terminal_node in terminal_nodes:
        index = terminal_node
        rules = np.array([])
        
        while index > 1:
            parent = np.floor(index / 2)
            if index % 2 == 0:
                rules = np.append(rules, "'x{:d} > {:.2f}'".format(int(node_features[parent]) + 1, node_splits[parent]))
            else:
                rules = np.append(rules, "'x{:d} <= {:.2f}'".format(int(node_features[parent]) + 1, node_splits[parent]))
            index = parent
        rules = np.flip(rules)
        rule_sets.append((terminal_node, rules, node_means[terminal_node]))
        print("Node {:02}: [{}] => {}".format(terminal_node, ' '.join(rules), node_means[terminal_node]))
    # your implementation ends above
    
P = 20
is_terminal, node_features, node_splits, node_means = decision_tree_regression_train(X_train, y_train, P)
y_interval_hat = decision_tree_regression_test(X_interval, is_terminal, node_features, node_splits, node_means)
fig = plot_figure(X_train, y_train, X_test, y_test, X_interval, y_interval_hat)
fig.savefig("decision_tree_regression_{}.pdf".format(P), bbox_inches = "tight")

y_train_hat = decision_tree_regression_test(X_train, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("RMSE on training set is {} when P is {}".format(rmse, P))

y_test_hat = decision_tree_regression_test(X_test, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("RMSE on test set is {} when P is {}".format(rmse, P))

P = 50
is_terminal, node_features, node_splits, node_means = decision_tree_regression_train(X_train, y_train, P)
y_interval_hat = decision_tree_regression_test(X_interval, is_terminal, node_features, node_splits, node_means)
fig = plot_figure(X_train, y_train, X_test, y_test, X_interval, y_interval_hat)
fig.savefig("decision_tree_regression_{}.pdf".format(P), bbox_inches = "tight")

y_train_hat = decision_tree_regression_test(X_train, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("RMSE on training set is {} when P is {}".format(rmse, P))

y_test_hat = decision_tree_regression_test(X_test, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("RMSE on test set is {} when P is {}".format(rmse, P))

extract_rule_sets(is_terminal, node_features, node_splits, node_means)
