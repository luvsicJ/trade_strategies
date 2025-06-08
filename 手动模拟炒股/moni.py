import random
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
df['盈亏百分比'] = 0.0  # 盈亏百分比列，默认为0
df = df.sort_values('日期').reset_index(drop=True)

# 默认起始日期和结束日期对应的下标
start_idx = random.randint(0, 1000)
end_idx = start_idx + 90
ori_money = 100000

# 初始化交易相关变量
portfolio = {
    'shares': 0,
    'portfolio_value': 0.0,
    'balance': ori_money,
    'ori': ori_money,
    'total_cost': 0.0,
    'avg_cost': 0.0,
}

# 创建 Dash 应用
app = dash.Dash(__name__)

# 创建 K 线图的函数
def create_candlestick_chart(data, avg_cost, closing_price, show_profit_line):
    fig = go.Figure()

    # 添加K线图
    fig.add_trace(go.Candlestick(
        x=data['日期'],
        open=data['开盘'],
        high=data['高'],
        low=data['低'],
        close=data['收盘'],
        increasing_line_color='red',
        decreasing_line_color='green',
        hovertext=data.apply(lambda row: (
            f"日期: {row['日期'].strftime('%Y-%m-%d') if pd.notnull(row['日期']) else '未知'}<br>"
            f"收盘价: {row['收盘']:.2f}<br>"
            f"涨跌金额: {(row['收盘'] - row['开盘']):.2f}<br>"
            f"涨跌幅: {row['涨跌幅']:.2f}%<br>"
            f"交易: {row['交易标记'] if row['交易标记'] else '无'}<br>"
            f"股价与成本差: {(row['收盘'] - avg_cost):.2f}元<br>"
            f"盈亏百分比: {row['盈亏百分比']:.2f}%"
        ), axis=1),
        hoverinfo="text",
        name="K线"
    ))

    # 添加盈亏百分比折线，初始根据 show_profit_line 决定是否显示
    fig.add_trace(go.Scatter(
        x=data['日期'],
        y=data['盈亏百分比'],
        mode='lines',
        name='盈亏百分比',
        line=dict(color='purple', width=2),
        yaxis='y2',
        visible=show_profit_line
    ))

    # 添加交易标记的 annotations
    annotations = []
    for idx, row in data.iterrows():
        if row['交易标记']:
            y_position = row['高'] + (row['高'] - row['低']) * 0.1
            annotations.append(dict(
                x=row['日期'],
                y=y_position,
                text=row['交易标记'],
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-30,
                font=dict(size=12, color='black'),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1
            ))

    # 添加平均成本线和标注
    if avg_cost > 0:
        fig.add_shape(
            type="line",
            x0=data['日期'].iloc[0],
            x1=data['日期'].iloc[-1],
            y0=avg_cost,
            y1=avg_cost,
            line=dict(
                color="#1E88E5",
                width=3,
                dash="dash"
            )
        )
        fig.add_annotation(
            x=data['日期'].iloc[-1],
            y=avg_cost,
            text=f"平均成本: {avg_cost:.2f}元",
            showarrow=False,
            font=dict(size=14, color="#1E88E5"),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#1E88E5',
            borderwidth=1,
            xanchor="left",
            xshift=10
        )

    # 获取当日的价格、涨跌幅、涨跌金额
    last_row = data.iloc[-1]
    current_date = last_row['日期'].strftime('%Y-%m-%d') if pd.notnull(last_row['日期']) else '未知'
    closing_price = last_row['收盘']
    price_change = last_row['收盘'] - last_row['开盘']
    percentage_change = last_row['涨跌幅']

    # 更新标题，包含当日的价格、涨跌幅和涨跌金额
    title_text = (
        f"{current_date} | "
        f"收盘价: {closing_price:.2f}元 | "
        f"涨跌: {price_change:.2f}元({percentage_change:.2f}%) "
    )

    # 更新布局，添加双Y轴
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor='center',
            y=0.95,
            yanchor='top',
            font=dict(size=16)
        ),
        xaxis_title='日期',
        yaxis_title='价格',
        yaxis2=dict(
            title='盈亏百分比 (%)',
            overlaying='y',
            side='right',
            showgrid=False,
            visible=show_profit_line  # 次坐标轴随折线显示/隐藏
        ),
        xaxis_rangeslider_visible=False,
        hovermode='closest',
        height=800,
        annotations=annotations
    )

    return fig

# 应用布局
app.layout = html.Div([
    html.Div([
        html.Label("输入交易股数:", style={'fontSize': 16, 'marginRight': 10}),
        dcc.Input(
            id="shares-input",
            type="number",
            value=100,
            min=1,
            style={
                'width': '100px',
                'height': '40px',
                'fontSize': 16,
                'marginRight': 10,
                'border': '1px solid #ccc',
                'borderRadius': 5
            }
        ),
        html.Button(
            "买入",
            id="buy-button",
            n_clicks=0,
            style={
                'width': '120px',
                'height': '40px',
                'fontSize': 16,
                'backgroundColor': '#4CAF50',
                'color': 'white',
                'border': 'none',
                'borderRadius': 5,
                'marginRight': 10,
                'cursor': 'pointer'
            }
        ),
        html.Button(
            "卖出",
            id="sell-button",
            n_clicks=0,
            style={
                'width': '120px',
                'height': '40px',
                'fontSize': 16,
                'backgroundColor': '#f44336',
                'color': 'white',
                'border': 'none',
                'borderRadius': 5,
                'marginRight': 10,
                'cursor': 'pointer'
            }
        ),
        html.Button(
            "下一天",
            id="next-day-button",
            n_clicks=0,
            style={
                'width': '120px',
                'height': '40px',
                'fontSize': 16,
                'backgroundColor': '#2196F3',
                'color': 'white',
                'border': 'none',
                'borderRadius': 5,
                'marginRight': 10,
                'cursor': 'pointer'
            }
        ),
        html.Button(
            "显示/隐藏盈亏折线",
            id="toggle-profit-line-button",
            n_clicks=0,
            style={
                'width': '160px',
                'height': '40px',
                'fontSize': 16,
                'backgroundColor': '#FF9800',
                'color': 'white',
                'border': 'none',
                'borderRadius': 5,
                'cursor': 'pointer'
            }
        ),
    ], style={
        'display': 'flex',
        'justifyContent': 'center',
        'alignItems': 'center',
        'margin': '20px 0'
    }),
    html.Div(id="portfolio-info", style={'textAlign': 'center', 'fontSize': 16, 'margin': '10px 0'}),
    dcc.Graph(id='k-line-chart'),
    html.Div(id="date-range", style={'textAlign': 'center', 'fontSize': 16, 'margin': '10px 0'}),
    dcc.Store(id='portfolio-store', data=portfolio),
    dcc.Store(id='df-store', data=df.to_dict('records')),
    dcc.Store(id='current-end-idx', data=end_idx),
    dcc.Store(id='profit-line-visibility', data=False)  # 存储折线显示状态，默认为隐藏
])

# 回调函数：更新图表、日期范围、投资组合信息和数据存储
@app.callback(
    [
        Output('k-line-chart', 'figure'),
        Output('date-range', 'children'),
        Output('portfolio-info', 'children'),
        Output('portfolio-store', 'data'),
        Output('df-store', 'data'),
        Output('current-end-idx', 'data'),
        Output('profit-line-visibility', 'data'),
        Output('toggle-profit-line-button', 'children')
    ],
    [
        Input('next-day-button', 'n_clicks'),
        Input('buy-button', 'n_clicks'),
        Input('sell-button', 'n_clicks'),
        Input('toggle-profit-line-button', 'n_clicks')
    ],
    [
        State('shares-input', 'value'),
        State('portfolio-store', 'data'),
        State('df-store', 'data'),
        State('current-end-idx', 'data'),
        State('profit-line-visibility', 'data')
    ]
)
def update_dashboard(next_clicks, buy_clicks, sell_clicks, toggle_clicks, shares_input, portfolio_data, df_data, current_end_idx, profit_line_visible):
    df_temp = pd.DataFrame(df_data)
    df_temp['日期'] = pd.to_datetime(df_temp['日期'], errors='coerce')
    portfolio = portfolio_data.copy()

    # 确定触发器
    trigger = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

    # 切换盈亏折线显示状态
    if trigger == 'toggle-profit-line-button':
        profit_line_visible = not profit_line_visible
    else:
        profit_line_visible = profit_line_visible

    # 更新当前结束下标
    if trigger == 'next-day-button':
        current_end_idx += 1
    elif trigger in ['buy-button', 'sell-button']:
        pass

    if current_end_idx > len(df_temp):
        current_end_idx = len(df_temp)

    data_to_show = df_temp.iloc[start_idx:current_end_idx]
    closing_price = df_temp.iloc[current_end_idx - 1]['收盘'] if current_end_idx <= len(df_temp) else 0

    if trigger == 'buy-button' and shares_input:
        cost = closing_price * shares_input
        if portfolio['balance'] >= cost:
            portfolio['shares'] += shares_input
            portfolio['total_cost'] += cost
            portfolio['balance'] -= cost
            current_idx = current_end_idx - 1
            df_temp.loc[current_idx, '交易标记'] = f"买入 {shares_input}股 @ {closing_price:.2f}"
            df_temp.loc[current_idx:, '当前持股'] = portfolio['shares']
            if portfolio['shares'] > 0:
                portfolio['avg_cost'] = portfolio['total_cost'] / portfolio['shares']

    elif trigger == 'sell-button' and shares_input:
        if portfolio['shares'] >= shares_input:
            portfolio['shares'] -= shares_input
            portfolio['total_cost'] -= portfolio['avg_cost'] * shares_input
            portfolio['balance'] += closing_price * shares_input
            current_idx = current_end_idx - 1
            df_temp.loc[current_idx, '交易标记'] = f"卖出 {shares_input}股 @ {closing_price:.2f}"
            df_temp.loc[current_idx:, '当前持股'] = portfolio['shares']
            if portfolio['shares'] == 0:
                portfolio['avg_cost'] = 0
                portfolio['total_cost'] = 0
            elif portfolio['shares'] > 0:
                portfolio['avg_cost'] = portfolio['total_cost'] / portfolio['shares']

    # 更新投资组合价值和每日盈亏百分比
    portfolio['portfolio_value'] = portfolio['shares'] * closing_price
    total_value = portfolio['balance'] + portfolio['portfolio_value']
    profit_percentage = ((total_value - portfolio['ori']) / portfolio['ori'] * 100) if portfolio['ori'] > 0 else 0
    current_idx = current_end_idx - 1
    df_temp.loc[current_idx, '盈亏百分比'] = profit_percentage

    # 创建图表
    figure = create_candlestick_chart(data_to_show, portfolio['avg_cost'], closing_price, profit_line_visible)

    date_range_display = (
        f"展示的日期范围: {data_to_show['日期'].iloc[0].strftime('%Y-%m-%d') if pd.notnull(data_to_show['日期'].iloc[0]) else '未知'} - "
        f"{data_to_show['日期'].iloc[-1].strftime('%Y-%m-%d') if pd.notnull(data_to_show['日期'].iloc[-1]) else '未知'}"
    )

    # 计算盈亏百分比（用于显示）
    portfolio_info = (
        f"持股数: {portfolio['shares']}股 | "
        f"平均成本: {portfolio['avg_cost']:.2f}元 | "
        f"盈亏:{total_value - portfolio['ori']}元  {profit_percentage:.2f}% | "
        f"证券价值: {portfolio['portfolio_value']:.2f}元 | "
        f"账户余额: {portfolio['balance']:.2f}元 "
    )

    # 更新按钮文本
    button_text = "显示盈亏折线" if not profit_line_visible else "隐藏盈亏折线"

    return figure, date_range_display, portfolio_info, portfolio, df_temp.to_dict('records'), current_end_idx, profit_line_visible, button_text

# 运行 Dash 应用
if __name__ == '__main__':
    app.run(debug=True, port=8051)