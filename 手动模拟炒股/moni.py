import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objects as go

# 读取 CSV 文件
csv_file_path = '/Users/apple/Desktop/MyProject/TradeStrategies_py/trade_strategies/历史数据/601888-中国中免-历史数据20100101～20250510.csv'
df = pd.read_csv(csv_file_path)

# 数据清洗
df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
price_columns = ['开盘', '收盘', '高', '低']
for col in price_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df['涨跌幅'] = pd.to_numeric(df['涨跌幅'].replace('[\%,]', '', regex=True), errors='coerce')
df.fillna({'涨跌幅': 0, '开盘': 0, '收盘': 0, '高': 0, '低': 0}, inplace=True)
df['交易标记'] = ''
df['当前持股'] = 0
df = df.sort_values('日期').reset_index(drop=True)

# 默认起始日期和结束日期对应的下标
start_idx = 2500
end_idx = start_idx + 30
ori_money = 100000


# 初始化交易相关变量
portfolio = {
    'shares': 0,
    'portfolio_value': 0.0,
    'balance': ori_money,
    'ori': ori_money,
}

# 创建 Dash 应用
app = dash.Dash(__name__)



# 创建 K 线图的函数
def create_candlestick_chart(data):
    fig = go.Figure(data=[go.Candlestick(
        x=data['日期'],
        open=data['开盘'],
        high=data['高'],
        low=data['低'],
        close=data['收盘'],
        increasing_line_color='red',
        decreasing_line_color='green',
        hovertext=data.apply(lambda row: (
            f"日期: {row['日期'].strftime('%Y-%m-%d') if pd.notnull(row['日期']) else '未知'}<br>"
            f"开盘价: {row['开盘']:.2f}<br>"
            f"收盘价: {row['收盘']:.2f}<br>"
            f"最高价: {row['高']:.2f}<br>"
            f"最低价: {row['低']:.2f}<br>"
            f"涨跌金额: {(row['收盘'] - row['开盘']):.2f}<br>"
            f"涨跌幅: {row['涨跌幅']:.2f}%<br>"
            f"交易: {row['交易标记'] if row['交易标记'] else '无'}<br>"
            f"当前持股: {int(row['当前持股'])}股"
        ), axis=1),
        hoverinfo="text"
    )])

    # 添加交易标记的 annotations
    annotations = []
    for idx, row in data.iterrows():
        if row['交易标记']:
            # 标记显示在K线的最高价上方
            y_position = row['高'] + (row['高'] - row['低']) * 0.1  # 略高于最高价
            annotations.append(dict(
                x=row['日期'],
                y=y_position,
                text=row['交易标记'],
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-30,
                font=dict(size=12, color='black'),
                bgcolor='rgba(255, 255, 255, 0.8)',  # 白色背景，略透明
                bordercolor='black',
                borderwidth=1
            ))

    fig.update_layout(
        title='K线图',
        xaxis_title='日期',
        yaxis_title='价格',
        xaxis_rangeslider_visible=False,
        hovermode='closest',
        height=800,
        annotations=annotations  # 添加 annotations
    )

    return fig

# 应用布局
app.layout = html.Div([
    dcc.Graph(id='k-line-chart', figure=create_candlestick_chart(df.iloc[start_idx:start_idx+1])),
    html.Button("下一天", id="next-day-button", n_clicks=0),
    html.Div([
        html.Label("输入交易股数:"),
        dcc.Input(id="shares-input", type="number", value=100, min=1),
        html.Button("买入N股", id="buy-button", n_clicks=0),
        html.Button("卖出N股", id="sell-button", n_clicks=0),
    ], style={'margin': '10px'}),
    html.Div(id="date-range"),
    html.Div(id="portfolio-info"),
    dcc.Store(id='portfolio-store', data=portfolio),
    dcc.Store(id='df-store', data=df.to_dict('records'))
])

# 回调函数：更新图表、日期范围、投资组合信息和数据存储
@app.callback(
    [
        Output('k-line-chart', 'figure'),
        Output('date-range', 'children'),
        Output('portfolio-info', 'children'),
        Output('portfolio-store', 'data'),
        Output('df-store', 'data')
    ],
    [
        Input('next-day-button', 'n_clicks'),
        Input('buy-button', 'n_clicks'),
        Input('sell-button', 'n_clicks')
    ],
    [
        State('shares-input', 'value'),
        State('portfolio-store', 'data'),
        State('df-store', 'data')
    ]
)
def update_dashboard(next_clicks, buy_clicks, sell_clicks, shares_input, portfolio_data, df_data):
    df_temp = pd.DataFrame(df_data)
    df_temp['日期'] = pd.to_datetime(df_temp['日期'], errors='coerce')
    portfolio = portfolio_data.copy()

    current_end_idx = start_idx + next_clicks + 1
    if current_end_idx > len(df_temp):
        current_end_idx = len(df_temp)
    data_to_show = df_temp.iloc[start_idx:current_end_idx]
    closing_price = df_temp.iloc[current_end_idx - 1]['收盘'] if current_end_idx <= len(df_temp) else 0

    trigger = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

    if trigger == 'buy-button' and shares_input:
        cost = closing_price * shares_input
        if portfolio['balance'] >= cost:
            portfolio['shares'] += shares_input
            portfolio['balance'] -= cost
            df_temp.loc[current_end_idx - 1, '交易标记'] += f"买入 {shares_input}股 @ {closing_price:.2f}; "
            df_temp.loc[current_end_idx - 1:, '当前持股'] = portfolio['shares']

    elif trigger == 'sell-button' and shares_input:
        if portfolio['shares'] >= shares_input:
            portfolio['shares'] -= shares_input
            portfolio['balance'] += closing_price * shares_input
            df_temp.loc[current_end_idx - 1, '交易标记'] += f"卖出 {shares_input}股 @ {closing_price:.2f}; "
            df_temp.loc[current_end_idx - 1:, '当前持股'] = portfolio['shares']

    portfolio['portfolio_value'] = portfolio['shares'] * closing_price

    figure = create_candlestick_chart(data_to_show)

    date_range_display = (
        f"展示的日期范围: {data_to_show['日期'].iloc[0].strftime('%Y-%m-%d') if pd.notnull(data_to_show['日期'].iloc[0]) else '未知'} - "
        f"{data_to_show['日期'].iloc[-1].strftime('%Y-%m-%d') if pd.notnull(data_to_show['日期'].iloc[-1]) else '未知'}"
    )

    portfolio_info = (
        f"持股数: {portfolio['shares']}股 | "
        f"证券价值: {portfolio['portfolio_value']:.2f}元 | "
        f"账户余额: {portfolio['balance']:.2f}元 | "
        f"盈亏金额: {(portfolio['balance'] + portfolio['portfolio_value'] - portfolio['ori']):.2f}元"
    )

    return figure, date_range_display, portfolio_info, portfolio, df_temp.to_dict('records')

# 运行 Dash 应用
if __name__ == '__main__':
    app.run(debug=True)
