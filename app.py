import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
import sys, os, time
from datetime import datetime, date
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate


# Since every subject has specific data and attributes we will use object oriented programming to define a subject as a class
class Subject:
    def __init__(self, series, csv):
        df = pd.read_csv(csv)
        df['time_new'] = pd.to_datetime(df['time_new'], format="%d%b%y:%H:%M:%S") # Format the datetime
        df.set_index("time_new", inplace=True)
        self.ident = series.iloc[0]
        self.age = series.iloc[1]
        self.BMI = series.iloc[2]
        self.weight = series.iloc[3]
        self.p_bodyfat = series.iloc[4]
        self.fatmass = series.iloc[5]
        self.fatfreemass = series.iloc[6]
        self.race = series.iloc[7]
        self.ethnicity = series.iloc[8]
        self.data = df
        # Calculate sleep hours by day within a 7PM - 12PM window
        sleep_df = df.resample('H').mean().between_time('19:00', '12:00')
        sleep = []
        day = []
        for i in range(len(sleep_df)):
            if (sleep_df.minute_mets[i] < 1.3 and sleep_df.ap_posture[i] == 0): # Criteria for sleep (minute_mets < 1.3 and ap_posture is 0)
                sleep.append(True)
                day.append(sleep_df.iloc[i].name.day)
            else:
                sleep.append(False)
        days_sleep = OrderedDict(Counter(day))
        for i, j in zip(range(len(sleep)), day):
            if i is not 0:
                if sleep[i-1] == True and sleep[i] == False and sleep[i+1] == True: # Smooth moments that abruptly change between wake and sleep 
                    sleep[i] = True
                    days_sleep[j] += 1
            
                if sleep[i-1] == False and sleep[i] == True and sleep[i+1] == False: # Smooth moments that abruptly change between sleep and wake
                    sleep[i] = False
                    days_sleep[j] -= 1
        self.sleep = days_sleep

dem_df = pd.read_csv("Data/demographics.csv") # Read demographics data
dem_df.drop(index=22, inplace=True) # Dropping subject 24 because no minute data is available for that subject
dem_df.reset_index(inplace=True, drop = True)
dem_df.id = dem_df.index

files = [] # Generate a list of all the minute by minute data for each patient
for filename in os.listdir("Data/"): # Data files in a sub directory named Data
    if filename.startswith("min") and filename.endswith(".csv"):
        files.append(filename)
files.sort()

subjects = [] # Generate subject objects with the above data
for i in range(len(dem_df)):
    subjects.append(Subject(dem_df.iloc[i], "Data/"+files[i]))

app = dash.Dash(__name__)
app.layout = html.Div(children = [
    html.H1(children = "Active Pal Sleep Analysis", style = {"textAlign":"center"}),
    dcc.Dropdown(id = 'ident', placeholder = 'Select a Subject', style = dict(marginBottom = '5px', width = '33%', display = 'inline-block'), options = [{'label': i, 'value':i} for i in range(len(dem_df))]),
    dcc.Dropdown(id = 'race', placeholder = 'Filter by Race', style = dict(marginBottom = '5px', width = '33%', display = 'inline-block'), options = [{'label': i, 'value':i} for i in dem_df.Race.unique()]),
    dcc.Dropdown(id = 'ethnicity', placeholder = 'Filter by Ethnicity', style = dict(width = '33%', display = 'inline-block'), options = [{'label':i, 'value':i} for i in dem_df.Ethnicity.unique()]),
    html.Div(id = "Graph-container", children = [dcc.Graph(id = "METs-container"), dcc.Graph(id = "Posture-container"), dcc.Graph(id = "Sleep-container"), dcc.Graph(id = "Seasonal-container")])
    
])

@app.callback(
    Output('ident', 'options'),
    [Input('race', 'value'),
     Input('ethnicity', 'value')]
    )
def filter_atts(race = 'value', ethnicity = 'value'):
    filt_result = []
    print(filt_result)
    if race is None and ethnicity is None:
        raise PreventUpdate
    if race is not None and ethnicity is None:
        for subject in subjects:
            if subject.race == race:
                filt_result.append(subject.ident)
    if ethnicity is not None and race is None:
        for subject in subjects:
            if subject.ethnicity == ethnicity:
                filt_result.append(subject.ident)
    if ethnicity is not None and race is not None:
        for subject in subjects:
            if subject.race == race and subject.ethnicity == ethnicity:
                filt_result.append(subject.ident)
    if sum(filt_result) < 1:
        return [{'label':'None available', 'value':'None'}]
    else:
        return [{'label':i, 'value':i} for i in filt_result]

@app.callback(
    Output('METs-container', 'figure'),
    [Input('ident', 'value')]
    )
def mets_chart(ident = 'value'):
    if ident is None:
        raise PreventUpdate
    fig = px.bar(subjects[ident].data['minute_mets'].resample('7D').sum(),
                 x=subjects[ident].data['minute_mets'].resample(
                     '7D').mean().index,
                 y='minute_mets',
                 labels={'x': 'Week of'})
    fig.update_xaxes(title="Week of ")
    fig.update_yaxes(title="Minute METs")
    fig.update_layout(title="Sums Minute METs by Week", title_x=0.5)
    return fig

@app.callback(
    Output('Posture-container', 'figure'),
    [Input('ident', 'value')]
    )
def post_chart(ident = 'value'):
    if ident is None:
        raise PreventUpdate
    posture_dict = {0:"Sedentary", 1:"Standing", 2:"Walking"}
    labels = subjects[ident].data.groupby(subjects[ident].data.index.week)['ap_posture'].value_counts().index
    labels = [(labels[i][0], posture_dict[labels[i][1]]) for i in range(len(labels))]
    fig = go.Figure(go.Bar(
        y=subjects[ident].data.groupby(subjects[ident].data.index.week)['ap_posture'].value_counts(),
        marker_color='indianred'
    ))
    fig.update_yaxes(title = "Counts")
    fig.update_xaxes(title = "ap_posture grouped by week")
    fig.update_layout(
        title = 'Counts of ap_posture by Week',
        title_x = 0.5,
        xaxis = dict(
            tickmode = 'array',
            tickvals = [i for i in range(len(subjects[ident].data.groupby(subjects[ident].data.index.week)['ap_posture'].value_counts().index))],
            ticktext = labels
        ))
    return fig

@app.callback(
    Output('Sleep-container', 'figure'),
    [Input('ident', 'value')]
    )
def sleep_chart(ident = 'value'):
    if ident is None:
        raise PreventUpdate
    weekly_sleep = []
    for i in range(len(list(subjects[ident].sleep.values()))//7):
        if i == 0:
            weekly_sleep.append((i, sum(list(subjects[ident].sleep.values())[i:i+7])))
        else:
            weekly_sleep.append((i, sum(list(subjects[ident].sleep.values())[(8*i):(((i+1)*8)-1)])))
    weekly_sleep
    fig = px.bar(weekly_sleep,
                 x = [weekly_sleep[i][0] for i in range(len(weekly_sleep))],
                 y = [weekly_sleep[i][1] for i in range(len(weekly_sleep))],
                 labels = {'y':'Hours Slept', 'x':'Week'})
    fig.update_xaxes(title="Week")
    fig.update_yaxes(title="Hours Slept Per Week")
    fig.update_layout(title="Sleep Hours per Week", title_x = 0.5)
    return fig

@app.callback(
    Output('Seasonal-container', 'figure'),
    [Input('ident', 'value')]
    )
def seasonal_chart(ident = 'value'):
    if ident is None:
        raise PreventUpdate
    result = seasonal_decompose(list(subjects[ident].data.minute_mets.resample('D').mean().dropna()), model='additive', freq = 7)

    df_observed = pd.DataFrame({'Value':result.observed})
    df_observed.index = ['Observation' for i in range(len(result.observed))]
    df_observed['type'] = df_observed.index
    df_observed.reset_index(inplace=True, drop=True)

    df_seasonal = pd.DataFrame({'Value':result.seasonal})
    df_seasonal.index = ['Seasonal' for i in range(len(result.seasonal))]
    df_seasonal['type'] = df_seasonal.index
    df_seasonal.reset_index(inplace=True, drop=True)

    df_trend = pd.DataFrame({'Value':result.trend})
    df_trend.index = ['Trend' for i in range(len(result.trend))]
    df_trend['type'] = df_trend.index
    df_trend.reset_index(inplace=True, drop=True)

    df_resid = pd.DataFrame({'Value':result.resid})
    df_resid.index = ['Residual' for i in range(len(result.resid))]
    df_resid['type'] = df_resid.index
    df_resid.reset_index(inplace=True, drop=True)

    frames = [df_observed, df_seasonal, df_trend, df_resid]

    line_df = pd.concat(frames, axis = 0)

    fig = px.line(line_df, x=line_df.index, y="Value", color="type", facet_col="type", labels = {'Value':"Minute METs"})
    fig.update_xaxes(title="Days")
    fig.update_layout(title="Seasonal Decomposition of Minute METs", title_x = 0.5)
    return fig

@app.callback(
    Output('Graph-container', 'style'),
    [Input('ident', 'value')]
    )
def hide_graph(value):
    if value is None:
        return {"visibility":"hidden"}
    else:
        time.sleep(1)
        return {"visibility": "visible"}

if __name__ == '__main__':
    app.run_server(debug=True)