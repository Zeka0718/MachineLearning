import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd


header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('MNIST_data/u.data', sep='\t', names=header)

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]

train_data, test_data = train_test_split(df, test_size=0.25)

train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1] - 1, line[2] - 1] = line[3]
test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]


user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
user_similarity = 1-user_similarity
item_similarity = 1-item_similarity

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array(
            [np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')


def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

print(str(rmse(user_prediction, test_data_matrix)))
print(str(rmse(item_prediction, test_data_matrix)))
print("..................................................................")

def gradient(features, label, a, cycle):

    n, m = np.shape(label)
    location = label.nonzero()
    i = location[0]
    j = location[1]
    x = np.random.random((n, features))
    weight = np.random.random((features, m))

    for o in range(cycle):
        x_t = x
        weight_t = weight
        for b in range(m):
            error = x[i[j==b], :].dot(weight[:, b])-label[i[j==b],b]
            weight_t[:, b] = weight_t[:, b]-a*x[i[j==b], :].T.dot(error)
        for c in range(n):
            error = x[c, :].dot(weight[:, j[i==c]])-label[c, j[i==c]]
            x_t[c, :] = x_t[c, :]-a*error.dot(weight[:, j[i==c]].T)
        x = x_t
        weight = weight_t
        print("the",o, "round")
    return x, weight

x, weight=gradient(10, train_data_matrix, 0.001, 100)
prediction=x.dot(weight)
print(str(rmse(prediction, test_data_matrix)))
print("..................................................................")
