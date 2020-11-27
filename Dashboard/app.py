import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import dash_table
import plotly.express as px
import pandas as pd
import pymongo
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from dash.dependencies import Input, Output
from flask import Flask
session = pymongo.MongoClient("mongodb://192.168.1.10:27017/")


def discrete_background_color_bins(df, n_bins=5, columns='all'):
    import colorlover
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    if columns == 'all':
        if 'id' in df:
            df_numeric_columns = df.select_dtypes('number').drop(['id'], axis=1)
        else:
            df_numeric_columns = df.select_dtypes('number')
    else:
        df_numeric_columns = df[columns]
    df_max = df_numeric_columns.max().max()
    df_min = df_numeric_columns.min().min()
    ranges = [
        ((df_max - df_min) * i) + df_min
        for i in bounds
    ]
    styles = []
    legend = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        backgroundColor = colorlover.scales[str(n_bins)]['div']['RdYlGn'][i - 1]
        color = 'white' if i > len(bounds) / 2. else 'inherit'

        for column in df_numeric_columns:
            styles.append({
                'if': {
                    'filter_query': (
                        '{{{column}}} >= {min_bound}' +
                        (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                    ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                    'column_id': column
                },
                'backgroundColor': backgroundColor,
                'color': color
            })
        legend.append(
            html.Div(style={'display': 'inline-block', 'width': '60px'}, children=[
                html.Div(
                    style={
                        'backgroundColor': backgroundColor,
                        'borderLeft': '1px rgb(50, 50, 50) solid',
                        'height': '10px'
                    }
                ),
                html.Small(round(min_bound, 2), style={'paddingLeft': '2px'})
            ])
        )

    return (styles, html.Div(legend, style={'padding': '5px 0 5px 0'}))

def apply_overlay(df, budget_control = True):
    moneys = list()
    predictions = list()
    
    for i in df.index:
        prediction = df.prediction[i]
        money = 10 
        overlay_prediction = np.where((prediction[0] > 0.60) & (prediction[2] < 0.05), np.array([1,0,0]), 
            np.where((prediction[1] > 0.60) & (prediction[2] < 0.05), np.array([0,1,0]), np.array([0,0,1])))
        moneys.append(money)
        predictions.append(overlay_prediction)
    
    if budget_control == True:
        return money, predictions
    
    else:
        return predictions
    
def fix_duplicates(df):
    datelist = list()
    
    def check(timestamp, datelist):
        if timestamp in datelist:
            timestamp = timestamp + pd.Timedelta("1s")
            return check(timestamp, datelist)
        else:
            return timestamp  
    
    for i in df.date:
        datelist.append(check(i, datelist))
        
    return datelist

def get_pnl(df, money, include_fees = False, commission = 0.05, odds_modifier = 0.0):
    pnl = list()
    if not isinstance(money, pd.Series):
        df["money"] = money
    for i in df.index:
        money = df.money[i]
        pred = df.prediction[i] 
        Y = df.result[i]
        odds_1 = df.odds_1[i] + odds_modifier
        odds_2 = df.odds_2[i] + odds_modifier
        if include_fees == False:
            pnl_game = -money + money*(pred[0]*Y[0])*odds_1 + money*(pred[1]*Y[2])*odds_2 + money*pred[2]
        else:
            profit_share = 1-commission
            pnl_game = -money + money*(pred[0]*Y[0])*odds_1*profit_share + money*(pred[1]*Y[2])*odds_2*profit_share + money*pred[2]
        pnl.append(pnl_game)
    return pnl

def run_pnl():
    #getting data
    db = session["processes"]
    cursor = db["test_csgo_bets"].find({})
    stats = list(cursor)
    df = pd.DataFrame(stats.copy())
    df = df[df.version == 2]
    
    db = session["betting"]
    cursor = db["match_results"].find({'h_id': {'$in': df.h_id.to_list()}})
    stats = list(cursor)
    results = pd.DataFrame(stats.copy())

    cursor = db["matches"].find({'h_id': {'$in': df.h_id.to_list()}})
    stats = list(cursor)
    dates = stats.copy()
    dates = pd.DataFrame(dates)

    #shaping data
    bets_made = len(df)
    df = df[df["h_id"].isin(results["h_id"])]
    dates = dates[dates["h_id"].isin(results["h_id"])]
    dates.set_index(["h_id"], inplace=True)
    results.result = results.result.apply(lambda x: np.array(x))
    results.set_index(["h_id"], inplace=True)
    df["money"], df.prediction = apply_overlay(df)
    df.prediction = df.prediction.apply(lambda x: np.array(x).round(0))
    df = df.merge(results[["team1","team2","result"]], on="h_id")
    df = df.merge(dates["date"], on="h_id")
    df["cf"] = get_pnl(df, df.money)
    df["cf_clean"] = get_pnl(df, df.money, include_fees = True, commission = 0.02)
    df["cf_synth"] = get_pnl(df, df.money, include_fees = True, commission = 0.02, odds_modifier = 0.05)
    df["fees"] =  df.cf_clean - df.cf 
    df["result_binary"] = df.result.apply(lambda x: 1 if x[0] == 1 else 2)
    df["pred_binary"] = df.prediction.apply(lambda x: 1 if x[0] == 1 else 2 if x[1] == 1 else 3)
    df["winner_odds"] = np.where(df.result_binary == 1, df.odds_1, df.odds_2)
    df["loser_odds"] = np.where(df.result_binary == 1, df.odds_2, df.odds_1)
    df["result_word"] = np.where((df["result_binary"] == df["pred_binary"]), "win", "loss")
    df["result_word"] = np.where((df["pred_binary"] == 3), "flat", df.result_word)
    df["date"] = pd.to_datetime(df.date)
    df["ts"] = pd.to_datetime(df.ts)
    df["date"] = df.ts.fillna(df.date)
    df["date"] = fix_duplicates(df)
    df = df.sort_values(by="date", ascending=False)
    df.index = df.date
    df = df.drop(columns=["date"])
    df = df.round(2)
    return df


def get_outstanding_bets():
    df = run_pnl()
    db = session["processes"]
    cursor = db["test_csgo_bets"].find({"version": {'$in': [2]}})
    stats = list(cursor)
    bets = pd.DataFrame(stats.copy())
    bets = bets[~bets["h_id"].isin(df["h_id"])]
    if bets.empty:
        return pd.DataFrame(columns=["Team 1", "Team 2", "Odds 1", "Odds 2", "N1", "N2", "N3"])
    
    else:
        db = session["betting"]
        cursor = db["matches"].find({'h_id': {'$in': bets.h_id.to_list()}})
        stats = list(cursor)
        dates = stats.copy()
        dates = pd.DataFrame(dates)
        
        dates.set_index(["h_id"], inplace=True)
        bets = bets.merge(dates[["date", "team1", "team2"]], on="h_id")
        bets.prediction = apply_overlay(bets, budget_control = False)
        bets["N1"] = bets.prediction.apply(lambda x: x[0])
        bets["N2"] = bets.prediction.apply(lambda x: x[1])
        bets["N3"] = bets.prediction.apply(lambda x: x[2])
        bets = bets.sort_values(ascending=False, by = "date")
        bets = bets[["team1", "team2", "odds_1", "odds_2", "N1", "N2", "N3"]].round(2)
        bets = bets.rename(columns={"team1" : "Team 1", "team2": "Team 2", "odds_1":"Odds 1", "odds_2":"Odds 2"})
        return bets


def get_return_table():
    df = run_pnl()
    df_return_table = df[["team1", "team2",  "result_word", "cf_synth", "fees"]]
    df_return_table = df_return_table.rename(columns={"team1": "Team 1", "team2": "Team 2", "result_word":"State", "cf_synth": "PnL","fees": "Fee"})
    return df_return_table

server = Flask(__name__)
app = dash.Dash(server = server)

app.layout = html.Div(
    children=[
        html.Div(className='row',
                 children=[
                    html.Div(className='four columns div-user-controls',
                             children=[
                                html.H3('Open Bets'),
                                dash_table.DataTable(id='open bets',
                                            columns=[{"name": i, "id": i} for i in get_outstanding_bets().columns],
                                            data=get_outstanding_bets().to_dict('records'),
                                            style_header={'backgroundColor': 'rgb(30, 30, 30)'},
                                            style_cell={
                                                'backgroundColor': 'rgb(50, 50, 50)',
                                                'color': 'white'
                                            },
                                            style_cell_conditional=[
                                            {
                                                'if': {'column_id': 'Team 1'},
                                                'textAlign': 'left'
                                            },
                                            {
                                                'if': {'column_id': 'Team 2'},
                                                'textAlign': 'left'
                                            },
                                            {
                                                'if': {'column_id': 'Odds 1'},
                                                'textAlign': 'center'
                                            },
                                            {
                                                'if': {'column_id': 'Odds 2'},
                                                'textAlign': 'center'
                                            },
                                            {
                                                'if': {'column_id': 'N1'},
                                                'textAlign': 'center'
                                            },
                                            {
                                                'if': {'column_id': 'N2'},
                                                'textAlign': 'center'
                                            },
                                            {
                                                'if': {'column_id': 'N3'},
                                                'textAlign': 'center'
                                            },],
                                            style_data_conditional=[
                                            {
                                                'if': {
                                                    'filter_query': '{N1} = 1',
                                                    'column_id': 'Odds 1'
                                                },
                                                'color': 'white',
                                                'fontWeight': 'bold',
                                                "backgroundColor": "#ff9514"
                                            },
                                            {
                                                'if': {
                                                    'filter_query': '{N2} = 1',
                                                    'column_id': 'Odds 2'
                                                },
                                                'color': 'white',
                                                'fontWeight': 'bold',
                                                "backgroundColor": "#ff9514"
                                            },
                                            {
                                                'if': {
                                                    'filter_query': '{N3} = 1',
                                                    'column_id': 'N3'
                                                },
                                                'color': 'white',
                                                'fontWeight': 'bold',
                                                "backgroundColor": "#ff9514"
                                            },]
                                        ),
                                 html.H3('Closed Bets'),
                                 dash_table.DataTable(id='closed bets',
                                            columns=[{"name": i, "id": i} for i in get_return_table().columns],
                                            data=get_return_table().to_dict('records'),
                                            style_header={'backgroundColor': 'rgb(30, 30, 30)'},
                                            style_cell={
                                                'backgroundColor': 'rgb(50, 50, 50)',
                                                'color': 'white'
                                            },
                                            style_cell_conditional=[
                                            {
                                                'if': {'column_id': 'Team 1'},
                                                'textAlign': 'left'
                                            },
                                            {
                                                'if': {'column_id': 'Team 2'},
                                                'textAlign': 'left'
                                            },
                                            {
                                                'if': {'column_id': 'State'},
                                                'textAlign': 'center'
                                            },
                                            {
                                                'if': {'column_id': 'PnL'},
                                                'textAlign': 'center'
                                            },
                                            {
                                                'if': {'column_id': 'Fee'},
                                                'textAlign': 'center'
                                            },],
                                            style_data_conditional=[
                                            {
                                                'if': {
                                                    'filter_query': '{PnL} < 0',
                                                    'column_id': 'PnL'
                                                },
                                                'color': 'white',
                                                "backgroundColor": "#B33B24"
                                            },
                                            {
                                                'if': {
                                                    'filter_query': '{PnL} = 0',
                                                    'column_id': 'PnL'
                                                },
                                                'color': '#9c824e',
                                                "backgroundColor": "#FCD667"
                                            },
                                            {
                                                'if': {
                                                    'filter_query': '{PnL} > 0 && {PnL} < 3',
                                                    'column_id': 'PnL'
                                                },
                                                'color': '#7BA05B',
                                                "backgroundColor": "#FFFF9F"
                                            },
                                            {
                                                'if': {
                                                    'filter_query': '{PnL} > 3 && {PnL} < 8',
                                                    'column_id': 'PnL'
                                                },
                                                'color': '#7BA05B',
                                                "backgroundColor": "#C5E17A"
                                            },
                                            {
                                                'if': {
                                                    'filter_query': '{PnL} > 8',
                                                    'column_id': 'PnL'
                                                },
                                                'color': 'white',
                                                "backgroundColor": "#5E8C31"
                                            },
                                            {
                                                'if': {
                                                    'filter_query': '{State} = "loss"',
                                                    'column_id': 'State'
                                                },
                                                'color': '#ffcccc',
                                                "backgroundColor": "#800000"
                                            },
                                            {
                                                'if': {
                                                    'filter_query': '{State} = "win"',
                                                    'column_id': 'State'
                                                },
                                                'color': '#ccff99',
                                                "backgroundColor": "#264d00"
                                            },
                                            {
                                                'if': {
                                                    'filter_query': '{State} = "flat"',
                                                    'column_id': 'State'
                                                },
                                                'color': '#fff5e6',
                                                "backgroundColor": "#b36b00"
                                            }]),
                                        
                                 dcc.Interval(id='interval-component-table', interval=60000, n_intervals=0)
                                ]
                             ),
                    html.Div(className='eight columns div-for-charts bg-grey',
                             children=[
                                    dcc.Graph(id='step-pnl', style={"height":"350px"}),
                                    dcc.Graph(id='daily-pnl', style={"height":"250px"}),
                                    dcc.Graph(id="odd-scatter", style={"height":"300px"}),
                                    dcc.Interval(id='interval-component-graph', interval=60000, n_intervals=0)
                             ])
                              ])
        ]

)

@app.callback(Output('odd-scatter','figure'),
            [Input('interval-component-graph','n_intervals')])
def get_pnl_scatter(n):
    df = run_pnl()
    wins = np.where(df.result_word == "win", df["winner_odds"], np.nan)
    loss = np.where(df.result_word == "loss", df["loser_odds"], np.nan)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=wins, mode='markers', name='Wins', marker=dict(color=('rgb(191, 255, 0)'))))
    fig.add_trace(go.Scatter(x=df.index, y=loss, mode='markers', name='Losses'))
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                                    'paper_bgcolor': 'rgba(0, 0, 0, 0)'},
                                    xaxis_tickformat = "%d %B")
    fig.update_layout(legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ))
    fig.update_layout(margin=dict(l=15, r=10, t=15, b=5), template = "plotly_dark")
    return fig

@app.callback(Output('step-pnl','figure'),
            [Input('interval-component-graph','n_intervals')])
def get_pnl_step(n):
    df = run_pnl()
    df_step = df.sort_index(ascending=True)
    fig_step_pnl = go.Figure()
    fig_step_pnl.add_trace(go.Scatter(x=df_step.index, y=df_step.cf.cumsum(), name="Raw CF", line_shape='vhv', 
                                        line = dict(color = ('rgb(7, 110, 227)'), width = 2)))
    fig_step_pnl.add_trace(go.Scatter(x=df_step.index, y=df_step.cf_clean.cumsum(), name="Clean CF", line_shape='vhv', 
                                        line = dict(color = ('rgb(105, 168, 240)'), width = 2)))
    fig_step_pnl.add_trace(go.Scatter(x=df_step.index, y=df_step.cf_synth.cumsum(), name="Synthetic CF", line_shape='vhv', 
                                        line = dict(color = ('rgb(211, 222, 235)'), width = 2)))
    fig_step_pnl.update_layout(template = "plotly_dark")
    fig_step_pnl.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                                        'paper_bgcolor': 'rgba(0, 0, 0, 0)'},
                                        xaxis_tickformat = "%d %B")
    fig_step_pnl.update_layout(legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ))
    fig_step_pnl.update_layout(
        margin=dict(l=15, r=10, t=15, b=5))

    fig_step_pnl.update_yaxes(tickprefix="€ ", showgrid=False)
    return fig_step_pnl

@app.callback(Output('daily-pnl','figure'),
            [Input('interval-component-graph','n_intervals')])
def get_daily_pnl(n):
    df = run_pnl()
    pnl = df["cf_synth"].resample("1D").sum()
    fig = px.bar(pnl, x=pnl.index, y="cf_synth", color="cf_synth",

                         template='plotly_dark', color_continuous_scale="Blues").update_layout(
                                   {'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                                    'paper_bgcolor': 'rgba(0, 0, 0, 0)'},
                                    xaxis_tickformat = "%d %B", margin=dict(l=15, r=10, t=15, b=5)).update(layout_coloraxis_showscale=False).update_yaxes( 
                                                                            tickprefix="€ ", showgrid=False)
    return fig


@app.callback(
        dash.dependencies.Output('open bets','data'),
        [dash.dependencies.Input('interval-component-table', 'n_intervals')])
def get_open_bets(n):
    return get_outstanding_bets().to_dict('records')

@app.callback(
        dash.dependencies.Output('closed bets','data'),
        [dash.dependencies.Input('interval-component-table', 'n_intervals')])
def get_closed_bets(n):
    return get_return_table().to_dict('records')


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port = 8081)
