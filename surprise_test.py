from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf
from surprise import Reader
import pandas as pd



# Load the movielens-100k dataset (download it if needed),
# and split it into 3 folds for cross-validation.
header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('MNIST_data/u.data', sep='\t', names=header)
df=df.drop('timestamp',1)


reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df=df, reader=reader)
data.split(n_folds=3)

# We'll use the famous SVD algorithm.
algo = SVD()

# Evaluate performances of our algorithm on the dataset.
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

print_perf(perf)