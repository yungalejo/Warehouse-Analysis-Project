#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from datetime import date, datetime, time, timedelta

import chart_studio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.offline as py
import seaborn as sns


# ## Import and Clean/Process Data

# In[13]:


all_data = pd.DataFrame()
df1 = pd.read_csv('data1.csv', parse_dates=[['Date', 'Time']])
df2 = pd.read_csv('data2.csv', parse_dates=[['Date', 'Time']])
df3 = pd.read_csv('data3.csv', parse_dates=[['Date', 'Time']])
df4 = pd.read_csv('data4.csv', parse_dates=[['Date', 'Time']])
all_data = pd.concat([df1, df2, df3, df4])
all_data.reset_index(drop=True, inplace=True)

all_data.columns


# In[14]:


all_data.drop(columns='Selector Name', inplace=True)
all_data.head()


# In[15]:


def convert_datetime(x):
    dt = str(x)
    t = dt[9:]
    year = int(dt[0:4])
    month = int(dt[4:6])
    day = int(dt[6:8])
    t1 = t.zfill(6)
    t2 = [t1[0]+t1[1], t1[2]+t1[3], t1[4] + t1[5]]
    t3 = [int(i) for i in t2]

    return datetime(year, month, day, t3[0], t3[1], t3[2])


# In[16]:


all_data['Date_Time'] = [convert_datetime(i) for i in all_data['Date_Time']]


# In[28]:


all_data['Hour'] = all_data['Date_Time'].dt.hour
all_data['Date'] = all_data['Date_Time'].dt.date
all_data.rename(columns={'QTY SHIP': 'Picked Quantity'}, inplace=True)
new_data = all_data.drop(columns='Date_Time')
new_data


# In[57]:


sums = new_data.groupby(['Selector ID', 'Date', 'Hour']).sum()
sums


# In[58]:


# Reset index so that each row represents data for a single hour on  a specific day.

sums = sums[['Picked Quantity']]
sums.reset_index(inplace=True)
sums.drop(labels=185, inplace=True)
sums


# In[59]:


sums['DateTime'] = pd.to_datetime(sums['Date'].apply(
    str)+' ' + sums['Hour'].apply(lambda x: str(x)+':00:00'))

sums['T/F'] = sums['Hour'].apply(lambda x: x >= 18 or x <= 6)
truesums = sums[(sums['T/F'] == True) & (sums['Picked Quantity'] < 300)]


# In[82]:


truesums


# ## Data Visualization

# In[54]:


hist = px.histogram(truesums, x="Picked Quantity", color_discrete_sequence=['#ff7f0e'],
                    title='Case per Hour Histogram and Boxplot', marginal="box", )

hist.update_layout(
    xaxis=dict(
        tickmode='linear',
        tick0=0,
        dtick=50,

    ))


hist.show()


# In[88]:


fig0 = px.scatter(sums, x='DateTime', y='Picked Quantity', color='Hour')
fig0.update_layout(title={
    'text': "Cases Per Hour",
    'y': .93,
    'x': 0.5,
    'xanchor': 'center',
    'yanchor': 'top'})
fig0.show()


# In[85]:


hours = sums[['Hour', 'Picked Quantity',
              'Selector ID', 'Date']].groupby('Hour').mean()
hours = hours.reset_index()

h1 = hours[:7]
h2 = hours[18:]
hours = pd.concat([h1, h2])
hours['avg'] = hours['Picked Quantity'].mean()


dates = sums[['Hour', 'Picked Quantity',
              'Selector ID', 'Date']].groupby('Date').mean()
dates = dates.reset_index()
dates.drop(labels=3, inplace=True)
dates['avg'] = dates['Picked Quantity'].mean()


# In[113]:


fig1 = go.Figure()

fig1.add_trace(go.Scatter(x=hours['Hour'], y=hours['Picked Quantity'],
                          mode='markers',
                          name='Average Cases/Hour',
                          marker=dict(size=8, color=hours['Hour'], colorscale='Viridis', colorbar=go.ColorBar(
                              title='Hour')),
                          showlegend=False
                          ))

fig1.add_trace(go.Scatter(x=hours['Hour'], y=hours['avg'],
                          mode='lines',
                          name='Total Average',
                          ))

fig1.update_layout(legend_orientation="h", title={
    'text': "Average Cases Per Hour",
    'y': .93,
    'x': 0.5,
    'xanchor': 'center',
    'yanchor': 'top'})

fig1.update_xaxes(title_text='Hour')
fig1.update_yaxes(title_text='Cases')

fig1.update_traces(marker=dict(size=12,
                               line=dict(width=2,
                                         color='DarkSlateGrey')),
                   selector=dict(mode='markers'))

fig1.show()


# In[108]:


fig2 = go.Figure()

fig2.add_trace(go.Scatter(x=dates['Date'], y=dates['Picked Quantity'],
                          mode='markers',
                          name='Average Cases/Hour',
                          showlegend=False,
                          marker=dict(size=8, colorscale='Viridis'),
                          ))

fig2.add_trace(go.Scatter(x=dates['Date'], y=dates['avg'],
                          mode='lines',
                          name='Total Average',
                          ))

fig2.update_layout(legend_orientation="h", title={
    'text': "Average Cases Per Hour by Day",
    'y': .93,
    'x': 0.5,
    'xanchor': 'center',
    'yanchor': 'top'})

fig2.update_xaxes(title_text='Date')
fig2.update_yaxes(title_text='Average Cases/Hour')

fig2.update_traces(marker=dict(size=12,
                               line=dict(width=2,
                                         color='DarkSlateGrey')),
                   selector=dict(mode='markers'))

fig2.show()


# #### Normality Assumption

# In[92]:


from matplotlib import pyplot
from matplotlib.pyplot import figure
from numpy.random import randn, seed
from statsmodels.graphics.gofplots import qqplot

# q-q plot

qqfig = qqplot(truesums['Picked Quantity'], line='s')
qqfig.set_size_inches(10, 5)
plt.title('Q-Q Plot')
pyplot.show()


# In[ ]:


# Save Graph Files
py.plot(hist, auto_open=False, filename='Histogram.html')
py.plot(fig0, auto_open=False, filename='CasesPerHour.html')
py.plot(fig1, auto_open=False, filename='AVGCasesPerHour.html')
py.plot(fig2, auto_open=False, filename='AVGCasesPerHour/Day.html')

