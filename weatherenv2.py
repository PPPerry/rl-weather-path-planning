import pandas as pd
from gridworld import *

weather = pd.read_csv('dataset/weather_final.csv')
weather = weather[(weather['hour'] == 3) & (weather['xid'] <= 95) & (weather['yid'] <= 173) & (weather['xid'] >= 56) & (weather['yid'] >= 134)]
weather.reset_index(drop=True, inplace=True)

x_max = int(weather['xid'].max() - weather['xid'].min()) + 1
y_max = int(weather['yid'].max() - weather['yid'].min()) + 1
x_min = int(weather['xid'].min())
y_min = int(weather['yid'].min())

def WeatherWorld():
    env = GridWorldEnv(n_width= x_max,
                       n_height = y_max,
                       u_size = 20,
                       default_reward = 0,
                       default_type = 0,
                       windy=False)
    env.start = (0, 39)
    env.ends = [(39, 0)]
    
    grid_matrix = np.zeros((x_max, y_max))
    obstacle = []
    # 障碍物势能场
    for row in weather.itertuples():
        if row.label == 1:
            print(int(row.xid), int(row.yid))
            env.types.append([int(row.xid) - x_min, int(row.yid) - y_min, 1])
            obstacle.append((int(row.xid) - x_min, int(row.yid) - y_min))

            if row.xid - x_min >= 2 and row.xid - x_min <= x_max - 3 and row.yid - y_min >= 2 and row.yid - y_min <= y_max - 3:
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        if i == row.xid - x_min and j == row.yid - y_min:
                            continue
                        grid_matrix[int(row.xid) - x_min + i, int(row.yid) - y_min + j] -= 2

    h, w = grid_matrix.shape
    # 目的地势能场
    for i in range(h):
        for j in range(w):
            if i == env.ends[0][0] and j == env.ends[0][1]:
                grid_matrix[i, j] = 10
                continue
            grid_matrix[i, j] += 5 / ((i - env.ends[0][0]) * (i - env.ends[0][0]) + (j - env.ends[0][1]) * (j - env.ends[0][1]))                

    for i in range(h):
        for j in range(w):
            if (i, j) in obstacle:
                continue
            else:
                env.rewards.append([i, j, grid_matrix[i, j]])
    env.rewards.append([1, 2, grid_matrix[i, j]])
    env.refresh_setting()
    # env.render()
    return env

env = WeatherWorld()