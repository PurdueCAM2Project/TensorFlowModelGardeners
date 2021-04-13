from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import pandas as pd

from NFold_GridSearch import NFold
from NFold_GridSearch import GridSearch

np.set_printoptions(threshold=np.inf)

print("> Preprocessing data...")

cancer_data = pd.read_csv(
    "breast-cancer-wisconsin.csv", delimiter=",", header=None)
cancer_data = cancer_data.replace("?", 0)
print("Total examples in dataset: %d" % (len(cancer_data)))

print("\n> Splitting data into 80% train and 20% test...")
# Splitting data into 20% test, 80% train
train, test = np.split(cancer_data.sample(frac=1), [int(.8 * len(cancer_data))])
train = np.array(train).astype("float")
train = train[:, 1:]
# Changing labels from 2 and 4 to -1 and 1
for row in train:
  if row[-1] == 2:
    row[-1] = -1
  else:
    row[-1] = 1

test = np.array(test).astype("float")
for row in test:
  if row[-1] == 2:
    row[-1] = -1
  else:
    row[-1] = 1
test_x = test[:, 1:-1]
test_y = test[:, -1]

print("Examples in training set: %d" % (len(train)))
print("Examples in test set: %d" % (len(test)))

# Using N-Fold validation
print("\n> Beginning N-Fold validation...")
n_fold = NFold(10)
n_fold.fit(train)
n_fold.predict()

# Now we will use our grid search to find the best combination of C values and penalties; 
# First, we split our training data into a training and "test" (validation) set 

train_x = train[:, :-1]
train_y = train[:, -1]
train_x_gs, test_x_gs, train_y_gs, test_y_gs = train_test_split(
    train_x, train_y, test_size=0.2, random_state=1, stratify=train_y)

# Next, we can go on to seeing which combination works best from our c_params and penalty lists
print("\n> Beginning Grid Search for C and Penalty hyperparameters...")
c_params = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
penalty_list = ["l1", "l2", "none"]
GS = GridSearch(c_params, penalty_list)
GS.fit(train_x_gs, train_y_gs)
GS.predict(test_x_gs, test_y_gs)

# After running the Grid Search a couple times, I've come to the conclusion that the
# best models are:
#   1. C: 0.1, penalty: l1
#   2. C: 1.0, penalty: l2;
# Let's go with C: 1.0, penalty: l2
print("\n> Testing our best model as predicted by grid search on test data:")
print("Model: C=1.0, penalty=l2")
lr = LogisticRegression(
    penalty="l2", C=1.0, random_state=1, solver="liblinear", multi_class="ovr")
lr.fit(train_x, train_y)
accuracy = lr.score(test_x, test_y)
print("Score: %.3f" % (accuracy))
