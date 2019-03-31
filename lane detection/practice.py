import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_excel("MLdataset.xlsx")
print(dataset.shape)
print(dataset.head())
now_time = datetime.now()
print(now_time.timestamp())
today = f'Year: {now_time.strftime("%Y")}, Month: {now_time.strftime("%m")}, Day: {now_time.strftime("%d")}, Hour: {now_time.strftime("%H")}, Minutes: {now_time.strftime("%M")}, Seconds: {now_time.strftime("%S")},Day of the week: {now_time.strftime("%w")}'
print(today)
# timestamp = dataset[:,1]
# timestamp

# random_device = np.random.randint(0,60)
# random_day = np.random.randint(0,30)
# day_find,day_lind = random_device*720+random_day*24,random_device*720 + random_day*24 + 23
# usage_random_day = dataset.iloc[day_find:day_lind,4]
# plt.plot(usage_random_day)
# plt.show()

def find_avg_usage(random_device):
    find,lind = random_device*720,random_device*720 + 719
    average_usage = np.zeros(shape=24)
    for i in range(0,24):
        usage_total = 0
        for j in range(find,lind):
            if(dataset.iloc[j,9])==i:
                usage_total += dataset.iloc[j,4]
        average_usage[i] = usage_total/24
    return average_usage

avg_usage_res = find_avg_usage(np.random.randint(0,20))
avg_usage_ind = find_avg_usage(np.random.randint(20,40))
avg_usage_com = find_avg_usage(np.random.randint(40,60))

# usage_random_day = dataset.iloc[find:lind,4]
# plt.plot(usage_random_day)
# plt.show()
print('hello')

plt.figure(figsize=(8,4))
ax = plt.subplot(111)
ax.bar(0,avg_usage_res)