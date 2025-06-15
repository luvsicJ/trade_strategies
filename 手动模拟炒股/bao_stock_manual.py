from dash import Dash, dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# 初始化 Dash 应用
app = Dash(__name__)

# 读取CSV文件
csv_file_path = '/Users/apple/Desktop/MyProject/TradeStrategies_py/trade_strategies/API数据获取/history_A_stock_k_data.csv'
df = pd.read_csv(csv_file_path)
df['deal'] = ''
df['pctChg_sum'] = 0.0  # 初始化pctChg_sum列

# 策略参数配置
初始资金 = 100000  # 初始资金10万元
pctChg_sum_N = 3

# 初始化账户变量
持仓股数 = 0
账户余额 = 初始资金
平均买入成本线 = 0

# 指定日期区间
current_date = '2021-06-11'
date_mask =  (df['date'] <= current_date)
filtered_df = df[date_mask]
current_idx = filtered_df.index[-1] if not filtered_df.empty else -1

def calculate_kdj(df, n=9, m1=3, m2=3, prefix=''):
    low_list = df['low'].rolling(n, min_periods=1).min()
    high_list = df['high'].rolling(n, min_periods=1).max()
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100
    rsv = rsv.fillna(50)
    K = np.full(len(df), 50.0)
    D = np.full(len(df), 50.0)
    for i in range(1, len(df)):
        K[i] = (2/3) * K[i-1] + (1/3) * rsv[i]
        D[i] = (2/3) * D[i-1] + (1/3) * K[i]
    J = 3 * K - 2 * D
    df[f'{prefix}K'] = K.round(2)
    df[f'{prefix}D'] = D.round(2)
    df[f'{prefix}J'] = J.round(2)
    return df

def init_all_j_values(df):
    df['date'] = pd.to_datetime(df['date'])
    df = calculate_kdj(df, prefix='daily_')
    weekly_df = df.resample('W-MON', on='date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    weekly_df = calculate_kdj(weekly_df, prefix='weekly_')
    monthly_df = df.resample('ME', on='date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    monthly_df = calculate_kdj(monthly_df, prefix='monthly_')
    all_dates = pd.DataFrame({'date': pd.date_range(
        start=df['date'].min(),
        end=df['date'].max(),
        freq='D'
    )})
    weekly_merged = all_dates.merge(
        weekly_df[['date', 'weekly_J']],
        on='date',
        how='left'
    )
    weekly_merged['week_tag'] = weekly_merged['date'].dt.to_period('W')
    weekly_merged['weekly_J'] = weekly_merged.groupby('week_tag')['weekly_J'].transform('last')
    monthly_merged = all_dates.merge(
        monthly_df[['date', 'monthly_J']],
        on='date',
        how='left'
    )
    monthly_merged['month_tag'] = monthly_merged['date'].dt.to_period('M')
    monthly_merged['monthly_J'] = monthly_merged.groupby('month_tag')['monthly_J'].transform('last')
    df = df.merge(weekly_merged[['date', 'weekly_J']], on='date', how='left')
    df = df.merge(monthly_merged[['date', 'monthly_J']], on='date', how='left')
    return df

# 初始化df J 值
df = init_all_j_values(df)

# 初始化全局变量
global_vars = {
    '持仓股数': 持仓股数,
    '账户余额': 账户余额,
    '平均买入成本线': 平均买入成本线,
    'current_idx': current_idx,
    'days_to_show': 30  # 默认显示前30天
}

def plot_kline_with_deals(df, start_idx, end_idx):
    plot_df = df.iloc[start_idx:end_idx+1].copy()
    fig = go.Figure()
    hovertext = [
        (
            f"日期: {date.strftime('%Y-%m-%d')}<br>"
            f"收盘价: {close:.2f}<br>"
            f"涨跌幅: {pctChg:.2f}%<br>"
            f"最近{pctChg_sum_N}日涨跌幅: {pctChg_sum:.2f}%<br>"
            f"日J: {daily_J:.2f}<br>"
            f"周J: {weekly_J:.2f}<br>"
            f"月J: {monthly_J:.2f}<br>"
            f"操作: {deal if deal else '无'}<br>"
            f"PE: {peTTM:.2f}<br>"
            f"PB: {pbMRQ:.2f}<br>"
        )
        for date, close, daily_J, weekly_J, monthly_J, deal, pctChg, pctChg_sum, peTTM, pbMRQ
        in zip(plot_df['date'], plot_df['close'], plot_df['daily_J'], plot_df['weekly_J'], plot_df['monthly_J'], plot_df['deal'],
               plot_df['pctChg'], plot_df['pctChg_sum'], plot_df['peTTM'], plot_df['pbMRQ'])
    ]
    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df['open'],
        high=plot_df['high'],
        low=plot_df['low'],
        close=plot_df['close'],
        increasing_line_color='red',
        decreasing_line_color='green',
        name='K线',
        yaxis='y1',
        hovertext=hovertext,
        hoverinfo='text'
    ))
    fig.add_trace(go.Scatter(
        x=plot_df.index,
        y=plot_df['盈亏百分比'],
        mode='lines',
        line=dict(color='purple', width=2),
        name='盈亏百分比(%)',
        yaxis='y2'
    ))
    for idx, row in plot_df.iterrows():
        if row['deal']:
            if '买入' in row['deal']:
                ay_direction = -40
                arrow_color = 'red'
            elif '卖出' in row['deal']:
                ay_direction = 40
                arrow_color = 'green'
            else:
                continue
            fig.add_annotation(
                x=idx,
                y=row['close'],
                text=row['deal'],
                showarrow=True,
                arrowhead=2,
                arrowcolor=arrow_color,
                ax=0,
                ay=ay_direction,
                xanchor='center',
                yanchor='middle',
                font=dict(size=10, color='black'),
                bgcolor='white',
                opacity=0.8
            )
    fig.update_layout(
        xaxis=dict(
            title='日期',
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            title='价格',
            side='left',
            showgrid=True
        ),
        yaxis2=dict(
            title='盈亏百分比(%)',
            side='right',
            overlaying='y',
            showgrid=False,
            zeroline=False
        ),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(family='Arial, sans-serif', size=12, color='#333')
    )
    if '平均买入成本线' in plot_df.columns:
        cost_line = plot_df['平均买入成本线'].where(plot_df['当前持有股数'] > 0)
        if not cost_line.isna().all():
            fig.add_trace(go.Scatter(
                x=cost_line.index,
                y=cost_line,
                mode='lines',
                line=dict(color='blue', width=1),
                name='成本线',
                yaxis='y1',
                connectgaps=False
            ))
    return fig

# Dash 布局
app.layout = html.Div([

    html.Div(id='current-kline-info', style={
        'margin': '5px',
        'padding': '5px',
        'border': '1px solid #d3d3d3',
        'borderRadius': '8px',
        'backgroundColor': '#f9f9f9',
        'display': 'flex',
        'justifyContent': 'space-around',
        'flexWrap': 'wrap',
        'fontSize': '14px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'fontFamily': 'Arial, sans-serif'
    }),
    html.Div([
        html.Label("股数:", style={'marginRight': '10px', 'fontSize': '14px'}),
        dcc.Input(id='shares-input', type='number', min=100, step=100, value=100, style={
            'marginRight': '10px',
            'padding': '5px',
            'borderRadius': '4px',
            'border': '1px solid #d3d3d3'
        }),
        html.Button('买入', id='buy-button', n_clicks=0, style={
            'marginRight': '10px',
            'padding': '8px 16px',
            'borderRadius': '4px',
            'border': 'none',
            'backgroundColor': '#28a745',
            'color': 'white',
            'cursor': 'pointer',
            'transition': 'background-color 0.2s'
        }),
        html.Button('卖出', id='sell-button', n_clicks=0, style={
            'marginRight': '10px',
            'padding': '8px 16px',
            'borderRadius': '4px',
            'border': 'none',
            'backgroundColor': '#dc3545',
            'color': 'white',
            'cursor': 'pointer',
            'transition': 'background-color 0.2s'
        }),
        html.Button('下一天', id='next-day-button', n_clicks=0, style={
            'marginRight': '10px',
            'padding': '8px 16px',
            'borderRadius': '4px',
            'border': 'none',
            'backgroundColor': '#007bff',
            'color': 'white',
            'cursor': 'pointer',
            'transition': 'background-color 0.2s'
        }),
        html.Label("显示前N天:", style={'marginRight': '10px', 'fontSize': '14px'}),
        dcc.Input(id='days-input', type='number', min=1, value=30, style={
            'marginRight': '5px',
            'padding': '5px',
            'borderRadius': '4px',
            'border': '1px solid #d3d3d3'
        }),
        html.Button('显示前N天数据', id='show-days-button', n_clicks=0, style={
            'padding': '8px 16px',
            'borderRadius': '4px',
            'border': 'none',
            'backgroundColor': '#6c757d',
            'color': 'white',
            'cursor': 'pointer',
            'transition': 'background-color 0.2s'
        }),
    dcc.Graph(id='kline-graph', style={'height': '70vh', 'margin': '20px'}),
    html.Div(id='account-info', style={
        'margin': '5px',
        'padding': '5px',
        'border': '1px solid #d3d3d3',
        'borderRadius': '8px',
        'backgroundColor': '#f9f9f9',
        'fontSize': '14px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'fontFamily': 'Arial, sans-serif'
    })
    ], style={'textAlign': 'center', 'marginTop': '20px'}),
], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '20px'})

# 回调函数
@app.callback(
    [Output('kline-graph', 'figure'),
     Output('account-info', 'children'),
     Output('current-kline-info', 'children')],
    [
        Input('buy-button', 'n_clicks'),
        Input('sell-button', 'n_clicks'),
        Input('next-day-button', 'n_clicks'),
        Input('show-days-button', 'n_clicks'),
        State('shares-input', 'value'),
        State('days-input', 'value')
    ]
)
def update_graph(buy_n_clicks, sell_n_clicks, next_day_n_clicks, show_days_n_clicks, shares, days):
    global global_vars, df
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

    # 处理显示天数
    if triggered_id == 'show-days-button' and days is not None:
        try:
            days = int(days)
            if days >= 1:
                global_vars['days_to_show'] = days
        except (ValueError, TypeError):
            global_vars['days_to_show'] = 30  # 无效输入时恢复默认值

    # 处理下一天
    if triggered_id == 'next-day-button' and global_vars['current_idx'] < len(df) - 1:
        global_vars['current_idx'] += 1

    idx = global_vars['current_idx']
    row = df.loc[idx]
    close_price = row.get('close', 0)
    daily_J = row.get('daily_J', 0)
    weekly_J = row.get('weekly_J', 0)
    monthly_J = row.get('monthly_J', 0)
    pctChg = row.get('pctChg', 0)
    peTTM = row.get('peTTM', 0)
    pbMRQ = row.get('pbMRQ', 0)
    deal = row.get('deal', '无')
    pctChg_sum = df.iloc[max(idx-pctChg_sum_N+1, 0):idx+1]['pctChg'].sum()
    df.at[idx, 'pctChg_sum'] = pctChg_sum

    # 计算PE和PB百分位
    historical_pe = df.iloc[:idx+1]['peTTM'].dropna()
    historical_pb = df.iloc[:idx+1]['pbMRQ'].dropna()
    if len(historical_pe) > 1:
        current_pe = historical_pe.iloc[-1]
        pe_exceed_count = (historical_pe < current_pe).sum()
        pe_percentile = 100 * pe_exceed_count / len(historical_pe)
    else:
        pe_percentile = 0
    if len(historical_pb) > 1:
        current_pb = historical_pb.iloc[-1]
        pb_exceed_count = (historical_pb < current_pb).sum()
        pb_percentile = 100 * pb_exceed_count / len(historical_pb)
    else:
        pb_percentile = 0

    # 处理买入
    if triggered_id == 'buy-button':
        可买数量 = int(shares) if shares else 0
        if 可买数量 >= 100:
            买入金额 = 可买数量 * close_price
            if 买入金额 <= global_vars['账户余额']:
                global_vars['账户余额'] -= 买入金额
                旧持仓价值 = global_vars['持仓股数'] * global_vars['平均买入成本线'] if global_vars['持仓股数'] > 0 else 0
                global_vars['持仓股数'] += 可买数量
                global_vars['平均买入成本线'] = (旧持仓价值 + 买入金额) / global_vars['持仓股数'] if global_vars['持仓股数'] > 0 else 0
                df.at[idx, 'deal'] = f'买入{可买数量}股@{close_price:.2f}'

    # 处理卖出
    elif triggered_id == 'sell-button':
        卖出数量 = int(shares) if shares else 0
        if global_vars['持仓股数'] >= 卖出数量 >= 100:
            卖出金额 = 卖出数量 * close_price
            global_vars['账户余额'] += 卖出金额
            global_vars['持仓股数'] -= 卖出数量
            global_vars['平均买入成本线'] = global_vars['平均买入成本线'] if global_vars['持仓股数'] > 0 else 0
            df.at[idx, 'deal'] = f'卖出{卖出数量}股@{close_price:.2f}'

    # 更新账户状态
    证券价值 = global_vars['持仓股数'] * close_price
    盈亏金额 = (证券价值 + global_vars['账户余额']) - 初始资金
    盈亏百分比 = (盈亏金额 / 初始资金) * 100 if 初始资金 != 0 else 0
    df.at[idx, '当前持有股数'] = global_vars['持仓股数']
    df.at[idx, '证券价值'] = 证券价值
    df.at[idx, '账户余额'] = global_vars['账户余额']
    df.at[idx, '盈亏金额'] = 盈亏金额
    df.at[idx, '盈亏百分比'] = 盈亏百分比
    df.at[idx, '平均买入成本线'] = global_vars['平均买入成本线'] if global_vars['持仓股数'] > 0 else 0

    # 生成当前K线信息
    kline_info = html.Div([
        html.Span(f"日期: {row['date'].strftime('%Y-%m-%d')}", style={'marginRight': '20px'}),
        html.Span(f"收盘价: {close_price:.2f}", style={'marginRight': '20px'}),
        html.Span(f"涨跌幅: {pctChg:.2f}%", style={'marginRight': '20px', 'color': 'green' if pctChg < 0 else 'red'}),
        html.Span(f"最近{pctChg_sum_N}日涨跌幅: {pctChg_sum:.2f}%", style={'marginRight': '20px', 'color': 'green' if pctChg_sum < 0 else 'red'}),
        html.Span(f"日J: {daily_J:.2f}", style={'marginRight': '20px', 'color': 'green' if daily_J < 0 else 'black'}),
        html.Span(f"周J: {weekly_J:.2f}", style={'marginRight': '20px', 'color': 'green' if weekly_J < 0 else 'black'}),
        html.Span(f"月J: {monthly_J:.2f}", style={'marginRight': '20px', 'color': 'green' if monthly_J < 0 else 'black'}),
        html.Span(f"PE: {peTTM:.2f} (百分位: {pe_percentile:.2f}%)", style={'marginRight': '20px', 'color': 'green' if pe_percentile < 30 else 'black'}),
        html.Span(f"PB: {pbMRQ:.2f} (百分位: {pb_percentile:.2f}%)", style={'marginRight': '20px', 'color': 'green' if pb_percentile < 30 else 'black'})
    ], style={'display': 'flex', 'flexWrap': 'wrap'})

    # 生成账户信息
    pct_color = 'red' if 盈亏百分比 < 0 else 'green'
    avg_cost_str = f"{global_vars['平均买入成本线']:.2f}" if global_vars['持仓股数'] > 0 else "无"
    account_info = html.Div([
        html.B("账户状态（最新）：", style={'marginRight': '20px'}),
        html.Span(f"当前持有股数: {global_vars['持仓股数']:.0f}", style={'marginRight': '20px'}),
        html.Span(f"证券价值: {证券价值:.2f}", style={'marginRight': '20px'}),
        html.Span(f"账户余额: {global_vars['账户余额']:.2f}", style={'marginRight': '20px'}),
        html.Span(f"盈亏金额: {盈亏金额:.2f}", style={'marginRight': '20px'}),
        html.Span(f"平均成本: {avg_cost_str}", style={'marginRight': '20px'}),
        html.Span(f"盈亏百分比: {盈亏百分比:.2f}%", style={'marginRight': '20px', 'color': 'green' if 盈亏百分比 < 0 else 'red'})
    ], style={
        'display': 'flex',
        'flexWrap': 'wrap',
        'padding': '15px',
        'border': '1px solid #d3d3d3',
        'borderRadius': '8px',
        'backgroundColor': '#f9f9f9',
        'fontSize': '14px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    })

    # 计算绘图的起始索引
    plot_start_idx = max(0, global_vars['current_idx'] - global_vars['days_to_show'])
    fig = plot_kline_with_deals(df, plot_start_idx, idx)

    return fig, account_info, kline_info

if __name__ == '__main__':
    app.run(debug=True)