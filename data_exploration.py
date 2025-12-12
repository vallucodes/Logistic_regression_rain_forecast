import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

raw_df = pd.read_csv('data/weatherAUS.csv')

print(raw_df)

raw_df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

# fig = px.histogram(raw_df,
#                    x='Location',
#                    title='Location vs. Rainy Days',
#                    color='RainToday')

# fig = px.histogram(raw_df,
#                    x='Temp9am',
#                    title='Temp at 3 pm vs. Rain Tomorrow',
#                    color='RainTomorrow')

# fig = px.histogram(raw_df,
#                     x='RainTomorrow',
#                     color='RainToday',
#                     title='Rain Tomorrow vs. Rain Today')

# fig = px.scatter(raw_df.sample(2000),
#                 title='Min Temp. vs Max Temp.',
#                 x='MinTemp',
#                 y='MaxTemp',
#                 color='RainToday')

# fig = px.scatter(raw_df.sample(2000),
#                 title='Temp (3 pm) vs. Humidity (3 pm)',
#                 x='Temp3pm',
#                 y='Humidity3pm',
#                 color='RainTomorrow')

# fig = px.histogram(raw_df.sample(2000),
#                 title='Rainfall',
#                 x='Rainfall',
#                 color='RainTomorrow')

# fig = px.histogram(raw_df.sample(2000),
#                 title='Evaporation',
#                 x='Evaporation',
#                 color='RainTomorrow')

# fig = px.histogram(raw_df.sample(2000),
#                 title='Sunshine',
#                 x='Sunshine',
#                 color='RainTomorrow')

# fig = px.histogram(raw_df.sample(2000),
#                 title='Humidity9am',
#                 x='Humidity9am',
#                 color='RainTomorrow')

# fig = px.histogram(raw_df.sample(2000),
#                 title='Humidity3pm',
#                 x='Humidity3pm',
#                 color='RainTomorrow')

# fig = px.histogram(raw_df.sample(2000),
#                 title='Pressure9am',
#                 x='Pressure9am',
#                 color='RainTomorrow')

# fig = px.histogram(raw_df.sample(2000),
#                 title='Pressure3pm',
#                 x='Pressure3pm',
#                 color='RainTomorrow')

# fig = px.histogram(raw_df.sample(2000),
#                 title='Cloud9am',
#                 x='Cloud9am',
#                 color='RainTomorrow')

# fig = px.histogram(raw_df.sample(2000),
#                 title='Cloud3pm',
#                 x='Cloud3pm',
#                 color='RainTomorrow')

# fig = px.histogram(raw_df.sample(2000),
#                 title='WindSpeed9am',
#                 x='WindSpeed9am',
#                 color='RainTomorrow',
#                 color_discrete_map={
#                     'No': '#db1818',
#                     'Yes': '#55b41d'
#                 })

# fig = px.histogram(raw_df.sample(2000),
#                 title='WindSpeed3pm',
#                 x='WindSpeed3pm',
#                 color='RainTomorrow',
#                 color_discrete_map={
#                     'No': '#db1818',
#                     'Yes': '#55b41d'
#                 })

# fig = px.histogram(raw_df.sample(2000),
#                 title='WindDir9am',
#                 x='WindDir9am',
#                 color='RainTomorrow',
#                 color_discrete_map={
#                     'No': '#db1818',
#                     'Yes': '#55b41d'
#                 })

# fig = px.histogram(raw_df.sample(2000),
#                 title='WindDir3pm',
#                 x='WindDir3pm',
#                 color='RainTomorrow',
#                 color_discrete_map={
#                     'No': '#db1818',
#                     'Yes': '#55b41d'
#                 })

# fig = px.histogram(raw_df.sample(2000),
#                 title='WindGustSpeed',
#                 x='WindGustSpeed',
#                 color='RainTomorrow',
#                 color_discrete_map={
#                     'No': '#db1818',
#                     'Yes': '#55b41d'
#                 })

# fig = px.histogram(raw_df.sample(2000),
#                 title='WindGustDir',
#                 x='WindGustDir',
#                 color='RainTomorrow',
#                 color_discrete_map={
#                     'No': '#db1818',
#                     'Yes': '#55b41d'
#                 })

# dirs = [
#     'N','NNE','NE','ENE',
#     'E','ESE','SE','SSE',
#     'S','SSW','SW','WSW',
#     'W','WNW','NW','NNW'
# ]

# # Aggregate count per (direction, RainTomorrow)
# df_counts = (
#     raw_df
#     .groupby(['WindGustDir', 'RainTomorrow'])
#     .size()
#     .reset_index(name='count')
# )

# fig = px.bar_polar(
#     df_counts,
#     r='count',
#     theta='WindGustDir',
#     color='RainTomorrow',
#     category_orders={'WindGustDir': dirs},
#     color_discrete_map={
#         'No': '#db1818',
#         'Yes': '#55b41d'
#     },
#     title="Wind Gust Direction Distribution by RainTomorrow",
#     direction="clockwise",
#     start_angle=90
# )

fig = px.scatter(raw_df.sample(2000),
                title='Humidity3pm vs. Pressure3pm',
                x='Humidity3pm',
                y='Pressure3pm',
                color='RainTomorrow')

fig.show()

