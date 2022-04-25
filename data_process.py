from os import _wrap_close
import pandas as pd

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# weather = pd.read_csv('dataset/ForecastDataforTraining_201802.csv')
# weather = weather[(weather['xid'] <= 100) & (weather['yid'] <= 100)]
# weather.reset_index(drop=True, inplace=True)

# weather_new = pd.DataFrame(columns=weather.columns)

# i = 0
# while i < weather.shape[0]:
#     weather_new = weather_new.append(weather.loc[i])
#     weather_new.loc[i//10, 'wind'] = weather.loc[i:i+9, 'wind'].mean()
#     weather_new.loc[i//10, 'rainfall'] = weather.loc[i:i+9, 'rainfall'].mean()
#     print(i)
#     i += 10


# weather_new = weather_new.drop('model', axis=1)

# weather_new.to_csv('dataset/weather_new_10.csv')

# weather = pd.read_csv('dataset/weather_new_10.csv')

# weather_new = pd.DataFrame(columns=weather.columns)
# i = 0
# j = 0
# while i < weather.shape[0]:
#     weather_new = weather_new.append(weather.loc[i])
#     weather_new.reset_index(drop=True, inplace=True)
#     weather_new.loc[j, 'wind'] = weather.loc[i:i+9, 'wind'].mean()
#     weather_new.loc[j, 'rainfall'] = weather.loc[i:i+9, 'rainfall'].mean()
#     weather_new.loc[j, 'xid'] = (weather.loc[i, 'xid'] - 1) // 10
#     weather_new.loc[j, 'yid'] = (weather.loc[i, 'yid'] - 1) // 10
#     print(i)
#     i += 10
#     j += 1

# weather_new.to_csv('dataset/weather_new_10_2.csv')

# weather = pd.read_csv('dataset/weather_new_10_2.csv')
# weather = weather[weather['hour'] == 3]
# weather.reset_index(drop=True, inplace=True)

# for row in weather.itertuples():
#     # if row.wind >= 15 or row.rainfall >= 4:
#     #     env.types.append([int(row.xid), int(row.yid), 1])
#     if row.wind <= 12 and row.rainfall <=2:
#         continue
#     else:
#         print(row.xid, row.yid, -(int(row.wind - 11) + int(row.rainfall - 1)))

# weather = pd.read_csv('dataset/weather_new_10_2.csv')

# weather_new = pd.DataFrame(columns=weather.columns)
# i = 0
# j = 0
# while i < weather.shape[0]:
#     weather_new = weather_new.append(weather.loc[i])
#     weather_new.reset_index(drop=True, inplace=True)
#     weather_new.loc[j, 'wind'] = weather.loc[i:i+9, 'wind'].mean()
#     weather_new.loc[j, 'rainfall'] = weather.loc[i:i+9, 'rainfall'].mean()
#     print(i)
#     i += 10
#     j += 1

# weather_new.to_csv('dataset/weather_new_10_final.csv')

option = '1'

if option == 'test':
    weather = pd.read_csv('dataset/ForecastDataforTesting_201802.csv')
    

    weather_test = pd.DataFrame(columns=['xid', 'yid', 'date_id', 'hour', 'wind1', 'wind2','wind3','wind4','wind5','wind6','wind7','wind8','wind9','wind10','rainfall1','rainfall2','rainfall3','rainfall4','rainfall5','rainfall6','rainfall7','rainfall8','rainfall9','rainfall10'])

    i = 0
    j = -1
    for i in range(weather.shape[0] // 500):
        if i % 10 == 0:
            print(i)
            j = j + 1
            weather_test.loc[j, 'xid'] = weather.loc[i, 'xid']
            weather_test.loc[j, 'yid'] = weather.loc[i, 'yid']
            weather_test.loc[j, 'date_id'] = weather.loc[i, 'date_id']
            weather_test.loc[j, 'hour'] = weather.loc[i, 'hour']
        wind = 'wind' + str(int(weather.loc[i, 'model']))
        rainfall = 'rainfall' + str(int(weather.loc[i, 'model']))
        weather_test.loc[j, wind] = weather.loc[i, 'wind']
        weather_test.loc[j, rainfall ] = weather.loc[i, 'rainfall']

    weather_test.to_csv('dataset/test.csv')





        


elif option == '1':
    

    weather = pd.read_csv('dataset/ForecastDataforTesting_201802.csv')
    print(weather['hour'].value_counts())


