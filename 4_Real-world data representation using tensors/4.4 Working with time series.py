# 4.4 Working with time series

import torch
import numpy as np
import csv

# 4.4.1 Adding a time dimension
# bike dataset
dir_path = "E:/DeepLearningExperiment/Deep-Learning-with-PyTorch/dlwpt-code-master/data/p1ch4/bike-sharing-dataset/hour-fixed.csv"
bikes_numpy = np.loadtxt(dir_path,
                         dtype=np.float32,
                         delimiter=",",
                         skiprows=1, 
                         converters={1: lambda x: float(x[8: 10])})
bikes = torch.from_numpy(bikes_numpy)
bikes.size()

# 4.4.2 Shaping the data by time period
bikes.shape, bikes.stride()

# CxHxW -> (-1, 24, )
daily_bikes = bikes.view(-1, 24, bikes.shape[1])
daily_bikes.shape
daily_bikes.stride()

daily_bikes = daily_bikes.transpose(1, 2)
daily_bikes.shape, daily_bikes.stride()


# 4.4.3 Ready for training
first_day = bikes[:24].long()
weather_onehot = torch.zeros(first_day.shape[0], 4)
first_day[:,9]

weather_onehot.scatter_(dim=1,
                        index=first_day[:,9].unsqueeze(1).long() - 1,
                        value=1.0)

torch.cat((bikes[:24], weather_onehot), 1)[:1]

daily_weather_onehot = torch.zeros(daily_bikes.shape[0], 4,
                                   daily_bikes.shape[2])
daily_weather_onehot.shape

daily_weather_onehot.scatter_(1, daily_bikes[:,9,:].long().unsqueeze(1) - 1, 1.0)
daily_weather_onehot.shape

daily_bikes = torch.cat((daily_bikes, daily_weather_onehot), dim=1)

daily_bikes[:, 9, :] = (daily_bikes[:, 9, :] - 1.0) / 3.0

temp = daily_bikes[:, 10, :]
temp_min = torch.min(temp)
temp_max = torch.max(temp)
daily_bikes[:, 10, :] = ((daily_bikes[:, 10, :] - temp_min)
                         / (temp_max - temp_min))

temp = daily_bikes[:, 10, :]
daily_bikes[:, 10, :] = ((daily_bikes[:, 10, :] - torch.mean(temp))
                         / torch.std(temp))