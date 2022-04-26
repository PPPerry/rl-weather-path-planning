import pandas as pd
from gridworld import *

def WeatherWorld():
    env = GridWorldEnv(n_width=10,
                       n_height = 10,
                       u_size = 40,
                       default_reward = 0,
                       default_type = 0,
                       windy=False)
    env.start = (0,9)
    env.ends = [(9,0)]
    

    weather = pd.read_csv('dataset/weather_new_10_final.csv')
    weather = weather[weather['hour'] == 3]
    weather.reset_index(drop=True, inplace=True)

    env.types = [(1, 6, 1), (2, 6, 1),  (2, 7, 1), (3, 1, 1), (4, 1, 1), (5, 2, 1)]
    env.rewards = []

    for row in weather.itertuples():
        if row.wind >= 16 or row.rainfall >= 4:
            env.types.append([int(row.xid), int(row.yid), 1])
        if row.wind <= 12 and row.rainfall <=2:
            continue
        else:
            env.rewards.append([row.xid, row.yid, -(int(row.wind - 11) + int(row.rainfall - 1))])

    env.rewards.append([9, 0, 5])  # 终点
    env.refresh_setting()
    return env