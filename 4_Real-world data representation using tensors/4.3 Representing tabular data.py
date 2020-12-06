# 4.3 Representing tabular data
#  列表(表格)数据的表示

import numpy as np
import csv
import torch


# 4.3.1 Using a real-world dataset


# 4.3.2 Loading a wine data tensor
# wine dataset
wine_path = "E:/DeepLearningExperiment/Deep-Learning-with-PyTorch/dlwpt-code-master/data/p1ch4/tabular-wine/winequality-white.csv"
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, 
                        delimiter=";", skiprows=1)

wineq_numpy

col_list = next(csv.reader(open(wine_path), delimiter=";"))
wineq_numpy.shape, col_list

# convert the NumPy array to a PyTorch tensor
wineq = torch.from_numpy(wineq_numpy)

wineq.shape, wineq.dtype


# 4.3.3 Representing scores
data = wineq[:, : -1] # Select all rows and all columns except the last
data, data.size()

target = wineq[:, -1] # Select all rows and the last column
target, target.shape

target = wineq[:, -1].long()
target, target.shape


# 4.3.4 One-hot encoding
target_onehot = torch.zeros(target.shape[0], 10) # (4898, 10)
target_onehot.shape

target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
target_onehot.shape

target_unsqueezed = target.unsqueeze(1)
target_unsqueezed


# 4.3.5 When to categorize 归类
data_mean = torch.mean(data, dim=0)
data_mean

data_var = torch.var(data, dim=0)
data_var

data_normalized = (data - data_mean) / torch.sqrt(data_var)
data_normalized


# 4.3.6 Finding thresholds
bad_indexes = target <= 3
bad_indexes.shape, bad_indexes.dtype, bad_indexes.sum()

bad_data = data[bad_indexes]
bad_data.shape

bad_data = data[target <= 3]
mid_data = data[(target > 3) & (target < 7)]
good_data = data[target >= 7]
bad_mean = torch.mean(bad_data, dim=0)
mid_mean = torch.mean(mid_data, dim=0)
good_mean = torch.mean(good_data, dim=0)

for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
    print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))

total_sulfur_threshold = 141.83
total_sulfur_data = data[:,6]
predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)
predicted_indexes.shape, predicted_indexes.dtype, predicted_indexes.sum()

actual_indexes = target > 5
actual_indexes.shape, actual_indexes.dtype, actual_indexes.sum()

n_matches = torch.sum(actual_indexes & predicted_indexes).item() # .item() get value
n_matches
n_predicted = torch.sum(predicted_indexes).item()
n_predicted
n_actual = torch.sum(actual_indexes).item()
n_actual

n_matches, n_matches / n_predicted, n_matches / n_actual