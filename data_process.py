from os import _wrap_close
import pandas as pd

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

weather = pd.read_csv('dataset/ForecastDataforTraining_201802.csv')
weather = weather[(weather['xid'] <= 100) & (weather['yid'] <= 100)]
weather.reset_index(drop=True, inplace=True)

weather_new = pd.DataFrame(columns=weather.columns)

i = 0
while i < weather.shape[0]:
    weather_new = weather_new.append(weather.loc[i])
    weather_new.loc[i//10, 'wind'] = weather.loc[i:i+9, 'wind'].mean()
    weather_new.loc[i//10, 'rainfall'] = weather.loc[i:i+9, 'rainfall'].mean()
    print(i)
    i += 10


weather_new = weather_new.drop('model', axis=1)

weather_new.to_csv('dataset/weather_new_10.csv')


