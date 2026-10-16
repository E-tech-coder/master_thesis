1. First , I split the trajectory dataset based on different time points in the "Splitting.ipynb"
2. Then I clustered the training trajectories and added a "cluster" column in the "Clustering_Train_Val.ipynb". The each validation trajectory section are assigned with a specific cluster in this file too.
3. Third, I normalized the training and validation trajectories in the "Normalization.ipynb"
4. At last, the normalized input and output windows of training trajectories and validation trajectories were generated and cluster 1 were trained in "Train_LSTM.ipynb"

