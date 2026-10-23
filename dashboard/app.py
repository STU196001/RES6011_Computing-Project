import pandas as pd
import json
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pickle
import os
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from dash import dash_table

# ---- Load data once, at startup ----
df = pd.read_csv("data_processed\metro_trips_model_ready.csv", parse_dates=['date'])
coords = pd.read_csv("data_processed\station_coordinates.csv")

with open("data_processed\line_order.json") as f:
    line_order = json.load(f)

stations = sorted(df['metro_station'].unique())
print(f"Loaded {len(stations)} stations, {len(df)} rows")


sarima_models = {}
for st in stations:
    safe_name = st.replace('/', '_').replace(' ', '_')
    path = f"models/sarima/{safe_name}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            sarima_models[st] = pickle.load(f)

lstm_models = {}
lstm_scalers = {}
for st in stations:
    safe_name = st.replace('/', '_').replace(' ', '_')
    model_path = f"models/lstm/{safe_name}.keras"
    scaler_path = f"models/lstm/{safe_name}_scaler.pkl"
    if os.path.exists(model_path):
        lstm_models[st] = load_model(model_path)
        with open(scaler_path, "rb") as f:
            lstm_scalers[st] = pickle.load(f)

print(f"Loaded {len(sarima_models)} SARIMA models, {len(lstm_models)} LSTM models")

comparison_df = pd.read_csv("docs/sarima_vs_lstm_comparison.csv")
comparison_lookup = comparison_df.set_index('station').to_dict('index')

median_sarima_mape = comparison_df['MAPE_sarima'].median()
n_lstm_fitted = len(lstm_models)

median_lstm_mape = comparison_df['MAPE_lstm'].median()
avg_rmse_sarima = comparison_df['RMSE_sarima'].mean()

comparison_df['winner_computed'] = comparison_df.apply(
    lambda r: 'SARIMA' if pd.isna(r['MAPE_lstm']) or r['MAPE_sarima'] < r['MAPE_lstm'] else 'LSTM', axis=1
)
n_sarima_wins = (comparison_df['winner_computed'] == 'SARIMA').sum()
n_lstm_wins = (comparison_df['winner_computed'] == 'LSTM').sum()

comparison_df['abs_diff'] = (comparison_df['MAPE_sarima'] - comparison_df['MAPE_lstm']).abs()
top_differences = comparison_df.dropna(subset=['abs_diff']).nlargest(4, 'abs_diff')

# ---- App ----
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.config.suppress_callback_exceptions = True

app.layout = dbc.Container([
    html.H1("Dubai Metro Ridership Dashboard", className="mt-3 mb-0"),
    html.P("Explore historical trips and forecasts by station", className="text-muted"),
    dcc.Tabs(id="tabs", value="explorer", children=[
        dcc.Tab(label="Explorer", value="explorer"),
        dcc.Tab(label="Comparison", value="comparison"),
        dcc.Tab(label="Insights", value="insights"),
    ]),
    html.Div(id="tab-content", className="mt-3")
], fluid=True)

coords_lookup = coords.set_index('metro_station')[['latitude', 'longitude']].to_dict('index')

with open("data_processed\schematic_positions.json") as f:
    schematic_positions = json.load(f)

LINE_COLORS = {
    'Green': '#639922',
    'Red_trunk': '#D85A30',
    'Red_branch_JGE': '#7F77DD',
    'Red_branch_EXPO': '#7F77DD'
}
INTERCHANGE_STATIONS = {'Union Metro Station', 'BurJuman Metro Station'}

station_line_map = {}
for segment_name, station_list in line_order.items():
    color = LINE_COLORS[segment_name]
    for s in station_list:
        station_line_map[s] = color

# Label position per station, matching the validated prototype layout
TEXT_SIDE = {}
# Alternate top/bottom along each horizontal run so neighbouring labels never share a side
alternating = [
    # Green line, top row (west to east)
    ("Al Qiyadah Metro Station", "top center"), ("Stadium Metro Station", "bottom center"),
    ("Al Nahda Metro Station", "top center"), ("Dubai Airport Free Zone Metro Station", "bottom center"),
    ("Al Qusais Metro Station", "top center"), ("Etisalat Metro Station", "bottom center"),
    # Red line, row east of Union
    ("Al Rigga Metro Station", "top center"), ("Deira City Centre Metro Station", "bottom center"),
    ("GGICO Metro Station", "top center"), ("Airport Terminal 1 Metro Station", "bottom center"),
    ("Airport Terminal 3 Metro Station", "top center"), ("Emirates Metro Station", "bottom center"),
    ("Rashidiya Metro Station", "top center"),
    # Green branch east of BurJuman
    ("Oud Metha Metro Station", "top center"), ("Dubai Healthcare City Metro Station", "bottom center"),
    ("Al Jadaf Metro Station", "top center"), ("Creek Metro Station", "bottom center"),
    # Route 2020 row
    ("The Gardens Metro Station", "top center"), ("Discovery Gardens Metro Station", "bottom center"),
    ("Al Furjan Metro Station", "top center"),
]
for s, side in alternating:
    TEXT_SIDE[s] = side
for s in ["Abu Hail Metro Station","Abu Baker Al Siddique Metro Station","Salah Al Din Metro Station",
          "Union Metro Station","Baniyas Square Metro Station","Gold Souq Metro Station","Al Ras Metro Station",
          "Al Ghubaiba Metro Station","Al Fahidi Metro Station","BurJuman Metro Station","Jabal Ali Metro Station",
          "DMCC Metro Station"]:
    TEXT_SIDE[s] = "middle left"
# Everything else defaults to the right (the long vertical Red trunk + branches' end points)
DEFAULT_SIDE = "middle right"

def build_schematic_map(selected_station=None):
    fig = go.Figure()

    for segment_name, station_list in line_order.items():
        if segment_name == "Red_branch_EXPO":
            # Explicit right-angle bend: Jabal Ali -> down -> right -> down to DIP -> EXPO
            jx, jy = schematic_positions["Jabal Ali Metro Station"]
            dx, dy = schematic_positions["Dubai Investment Park Metro Station"]
            ex, ey = schematic_positions["EXPO Metro Station"]
            xs = [jx, jx, dx, dx, ex]
            ys = [jy, jy + 52, jy + 52, dy, ey]
        else:
            xs = [schematic_positions[s][0] for s in station_list]
            ys = [schematic_positions[s][1] for s in station_list]

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines',
            line=dict(width=5, color=LINE_COLORS[segment_name]),
            showlegend=False, hoverinfo='skip'
        ))

    names = list(schematic_positions.keys())
    xs = [schematic_positions[n][0] for n in names]
    ys = [schematic_positions[n][1] for n in names]
    colors = ['#EF9F27' if n in INTERCHANGE_STATIONS else station_line_map.get(n, '#999') for n in names]
    sizes = [16 if n in INTERCHANGE_STATIONS else 10 for n in names]
    line_widths = [3 if n == selected_station else 1 for n in names]
    text_positions = [TEXT_SIDE.get(n, DEFAULT_SIDE) for n in names]

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='markers+text',
        marker=dict(size=sizes, color=colors, line=dict(width=line_widths, color='#333')),
        text=[n.replace(' Metro Station', '') for n in names],
        textposition=text_positions, textfont=dict(size=8),
        customdata=names, hoverinfo='text', hovertext=names,
        showlegend=False, cliponaxis=False
    ))

    fig.update_layout(
        height=700, margin=dict(l=170, r=120, t=10, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False, autorange='reversed'),
        plot_bgcolor='white'
    )
    return fig

def stat_card(icon, title, value, accent=False):
    return dbc.Card(dbc.CardBody([
        html.Div(icon, style={'fontSize': '20px'}),
        html.Div(title, className="text-muted small mt-1"),
        html.Div(value, className="h5 mb-0", style={'color': '#1B3A5C'})
    ]), className="h-100", style={'borderLeft': '4px solid #17A2A0' if accent else '4px solid #dee2e6'})


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "explorer":
        return html.Div([
            dbc.Row([
                dbc.Col(stat_card("1","Total stations", "53", accent=True), width=3),
                dbc.Col(stat_card("2","Best model overall", f"SARIMA · {median_sarima_mape:.1f}% MAPE"), width=3),
                dbc.Col(stat_card("3","Validated data range", "Jan 2018 – Aug 2023"), width=3),
                dbc.Col(stat_card("4","LSTM models fitted", f"{n_lstm_fitted} / 53 stations"), width=3),
            ], className="mb-3 g-2"),

            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Network map — 53 stations, coloured by line"),
                    dbc.CardBody([
                        dcc.Graph(id='metro-map', figure=build_schematic_map()),
                        html.Div([
                            html.Span("● Red", style={'color': '#D85A30', 'marginRight': '12px'}),
                            html.Span("● Green", style={'color': '#639922', 'marginRight': '12px'}),
                            html.Span("● Route 2020", style={'color': '#7F77DD', 'marginRight': '12px'}),
                            html.Span("● Interchange", style={'color': '#EF9F27'}),
                        ], className="small mt-2")
                    ])
                ]), width=7),

                dbc.Col([
                    dbc.Card(dbc.CardBody([
                        dcc.Dropdown(
                            id='station-dropdown',
                            options=[{'label': s, 'value': s} for s in stations],
                            value='BurJuman Metro Station',
                            clearable=False
                        )
                    ]), className="mb-2"),

                    dbc.Card(dbc.CardBody([
                        dbc.RadioItems(
                            id='model-toggle',
                            options=[{'label': 'SARIMA', 'value': 'sarima'}, {'label': 'LSTM', 'value': 'lstm'}],
                            value='sarima', inline=True
                        ),
                        html.Div([
                            html.Span("Forecast horizon: ", className="text-muted small"),
                            html.Span(id='horizon-label', className="small fw-bold")
                        ], className="mt-2"),
                        dcc.Slider(id='horizon-slider', min=1, max=12, step=1, value=6,
                                   marks={1: '1 mo', 6: '6 mo', 12: '12 mo'})
                    ]), className="mb-2"),

                    html.Div(id='detail-panel')
                ], width=5)
            ])
        ])

    if tab == "comparison":
        box_fig = go.Figure()
        box_fig.add_trace(go.Box(y=comparison_df['MAPE_sarima'], name='SARIMA', marker_color='#2E5C8A'))
        box_fig.add_trace(go.Box(y=comparison_df['MAPE_lstm'].dropna(), name='LSTM', marker_color='#D85A30'))
        box_fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                               yaxis_title="MAPE (%)")

        diff_items = []
        for _, row in top_differences.iterrows():
            diff_items.append(html.Li(
                f"{row['station']}: SARIMA {row['MAPE_sarima']:.1f}% vs LSTM {row['MAPE_lstm']:.1f}% "
                f"({'LSTM better' if row['winner_computed']=='LSTM' else 'SARIMA better'})"
            ))

        table_data = comparison_df[['station', 'MAPE_sarima', 'MAPE_lstm', 'winner_computed']].copy()
        table_data['MAPE_sarima'] = table_data['MAPE_sarima'].round(1)
        table_data['MAPE_lstm'] = table_data['MAPE_lstm'].round(1)
        table_data = table_data.rename(columns={
            'station': 'Station', 'MAPE_sarima': 'SARIMA MAPE (%)',
            'MAPE_lstm': 'LSTM MAPE (%)', 'winner_computed': 'Winner'
        })

        return html.Div([
            dbc.Row([
                dbc.Col(stat_card("1","SARIMA median MAPE", f"{median_sarima_mape:.2f}%", accent=True), width=3),
                dbc.Col(stat_card("2","LSTM median MAPE", f"{median_lstm_mape:.2f}%"), width=3),
                dbc.Col(stat_card("3","Stations won", f"SARIMA {n_sarima_wins} · LSTM {n_lstm_wins}"), width=3),
                dbc.Col(stat_card("4","Avg RMSE (SARIMA)", f"{avg_rmse_sarima:,.0f}"), width=3),
            ], className="mb-3 g-2"),

            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("MAPE distribution (box plot)"),
                    dbc.CardBody(dcc.Graph(figure=box_fig, config={'displayModeBar': False}))
                ]), width=6),
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Top differences"),
                    dbc.CardBody(html.Ul(diff_items))
                ]), width=6),
            ], className="mb-3 g-2"),

            dbc.Card([
                dbc.CardHeader("Full results — sortable by column"),
                dbc.CardBody(dash_table.DataTable(
                    data=table_data.to_dict('records'),
                    columns=[{'name': c, 'id': c} for c in table_data.columns],
                    sort_action='native', page_size=15,
                    style_cell={'padding': '6px', 'fontFamily': 'sans-serif', 'fontSize': '13px'},
                    style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0'},
                    style_data_conditional=[
                        {'if': {'filter_query': '{Winner} = "LSTM"'}, 'backgroundColor': '#fdf3e7'}
                    ]
                ))
            ])
        ])

    if tab == "insights":
        system_total = df.groupby('date')['trips'].sum().reset_index()

        trend_fig = go.Figure()
        trend_fig.add_vrect(x0='2020-03-01', x1='2020-11-01', fillcolor='#f8d7da', opacity=0.5,
                             line_width=0, annotation_text="COVID dip", annotation_position="top left")
        trend_fig.add_trace(go.Scatter(x=system_total['date'], y=system_total['trips'],
                                         mode='lines', line=dict(color='#2E5C8A', width=2)))
        trend_fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)

        stable = df[(df['date'] < '2020-03-01') | (df['date'] >= '2022-01-01')].copy()
        stable = stable[stable['metro_station'] != 'EXPO Metro Station']  # known special-event outlier, see Ch.8
        stable['station_mean'] = stable.groupby('metro_station')['trips'].transform('mean')
        stable['normalized'] = stable['trips'] / stable['station_mean']
        stable['month_num'] = stable['date'].dt.month
        seasonal = stable.groupby('month_num')['normalized'].mean().reindex(range(1, 13))
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        swing = (seasonal.max() - seasonal.min()) * 100
        swing_text = f"~{swing:.0f}% swing — {month_names[int(seasonal.idxmax()) - 1]} peak, {month_names[int(seasonal.idxmin()) - 1]} trough"

        seasonal_fig = go.Figure()
        seasonal_fig.add_hline(y=1.0, line_dash='dash', line_color='#999')
        seasonal_fig.add_trace(go.Scatter(x=month_names, y=seasonal.values, mode='lines+markers',
                                            line=dict(color='#2E5C8A', width=2)))
        seasonal_fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                                     yaxis_title="Ratio to station average")

        return html.Div([
            dbc.Card([
                dbc.CardHeader("System-wide monthly trips, Jan 2018 – Aug 2023"),
                dbc.CardBody(dcc.Graph(figure=trend_fig, config={'displayModeBar': False}))
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Seasonal pattern (normalized, station average = 1.0)"),
                    dbc.CardBody([
                        dcc.Graph(figure=seasonal_fig, config={'displayModeBar': False}),
                        html.Div(swing_text, className="text-muted small")
                    ])
                ]), width=6),

                dbc.Col([
                    dbc.Alert([
                        html.Strong("EXPO Metro Station — "),
                        "extreme spike in Mar 2022 (Expo 2020 closing month). Special-event outlier, not a data error."
                    ], color="warning", className="mb-2"),
                    dbc.Alert([
                        html.Strong("Route 2020 stations — "),
                        "distinct ramp-up pattern from 2020 opening, overlapping COVID-19 suppression."
                    ], color="info", className="mb-2"),
                    dbc.Alert([
                        html.Strong("World Trade Centre — "),
                        "high month-to-month volatility, not well captured by either model (SARIMA 10.7%, LSTM 21.3% MAPE)."
                    ], color="secondary"),
                ], width=6)
            ])
        ])

    return html.Div(f"Tab selected: {tab}")
    

@app.callback(
    Output('station-dropdown', 'value'),
    Input('metro-map', 'clickData'),
    prevent_initial_call=True
)
def map_click_to_dropdown(clickData):
    if clickData is None:
        return dash.no_update
    point = clickData['points'][0]
    if 'customdata' not in point:
        return dash.no_update
    return point['customdata']


def make_sequences(data, window=12):
    X = []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
    return np.array(X)

@app.callback(
    Output('detail-panel', 'children'),
    Output('metro-map', 'figure'),
    Input('station-dropdown', 'value'),
    Input('model-toggle', 'value'),
    Input('horizon-slider', 'value')
)
def update_detail_panel(selected_station, model_choice, horizon):
    fig = build_schematic_map(selected_station=selected_station)

    series = df[df['metro_station'] == selected_station].set_index('date')['trips'].asfreq('MS')
    has_lstm = selected_station in lstm_models

    chart_fig = go.Figure()
    chart_fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines',
                                     name='Historical', line=dict(color='#666')))

    if model_choice == 'lstm' and not has_lstm:
        model_choice = 'sarima'  # graceful fallback, NFR3

    if model_choice == 'sarima':
        fitted = sarima_models[selected_station]
        fc = fitted.get_forecast(steps=horizon)
        pred = fc.predicted_mean
        ci = fc.conf_int()
        future_dates = pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=horizon, freq='MS')
        chart_fig.add_trace(go.Scatter(
            x=future_dates, y=pred.values, mode='lines+markers',
            name='SARIMA forecast', line=dict(color='#2E5C8A', dash='dash'), marker=dict(size=7),
            error_y=dict(type='data', symmetric=False,
                array=(ci.iloc[:, 1].values - pred.values),
                arrayminus=(pred.values - ci.iloc[:, 0].values),
                visible=(horizon == 1))))      # interval bars only needed when the band collapses to one month
        chart_fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(ci.iloc[:, 1]) + list(ci.iloc[:, 0][::-1]),
            mode='lines', fill='toself', fillcolor='rgba(46,92,138,0.15)', line=dict(width=0),
            name='95% CI', showlegend=False))
    else:
        model = lstm_models[selected_station]
        scaler = lstm_scalers[selected_station]
        scaled = scaler.transform(series.values.reshape(-1, 1))
        window = 12
        last_window = scaled[-window:].reshape(1, window, 1)
        preds_scaled = []
        current_window = last_window.copy()
        for _ in range(horizon):
            next_val = model.predict(current_window, verbose=0)
            preds_scaled.append(next_val[0, 0])
            current_window = np.append(current_window[:, 1:, :], [[next_val[0]]], axis=1)
        pred = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
        future_dates = pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=horizon, freq='MS')
        chart_fig.add_trace(go.Scatter(x=future_dates, y=pred, mode='lines+markers',
                            name='LSTM forecast', line=dict(color='#D85A30', dash='dash'),
                            marker=dict(size=7)))

    chart_fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10),
                              legend=dict(orientation='h', y=-0.2))

    comp = comparison_lookup.get(selected_station, {})
    mape_sarima = comp.get('MAPE_sarima')
    mape_lstm = comp.get('MAPE_lstm')

    if not has_lstm:
        badge = dbc.Alert("LSTM unavailable for this station (insufficient history) — showing SARIMA", color="warning")
    elif mape_sarima is not None and mape_lstm is not None:
        winner = 'SARIMA' if mape_sarima < mape_lstm else 'LSTM'
        shown = model_choice.upper()
        if shown == winner:
            badge = dbc.Alert(f"{winner} recommended for this station ({mape_sarima:.1f}% vs {mape_lstm:.1f}% MAPE)", color="success")
        else:
            badge = dbc.Alert(f"Showing {shown} — {winner} performs better here ({mape_sarima:.1f}% vs {mape_lstm:.1f}% MAPE)", color="warning")
    else:
        badge = html.Div()

    panel = html.Div([
        dbc.Card([
            dbc.CardHeader(f"{selected_station} — historical + forecast"),
            dbc.CardBody(dcc.Graph(figure=chart_fig, config={'displayModeBar': False}))
        ], className="mb-2"),
        badge
    ])
    return panel, fig

@app.callback(
    Output('model-toggle', 'options'),
    Output('model-toggle', 'value'),
    Input('station-dropdown', 'value'),
    Input('model-toggle', 'value')
)
def update_model_toggle_options(selected_station, current_value):
    has_lstm = selected_station in lstm_models
    options = [
        {'label': 'SARIMA', 'value': 'sarima'},
        {'label': 'LSTM', 'value': 'lstm', 'disabled': not has_lstm}
    ]
    value = current_value if (current_value == 'sarima' or has_lstm) else 'sarima'
    return options, value

@app.callback(
    Output('horizon-label', 'children'),
    Input('horizon-slider', 'value')
)
def update_horizon_label(horizon):
    return f"{horizon} month{'s' if horizon != 1 else ''}"

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)