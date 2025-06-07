import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go

# 读取 CSV 文件
csv_file_path = '/Users/apple/Desktop/MyProject/TradeStrategies_py/trade_strategies/历史数据/601888-中国中免-历史数据20100101～20250510.csv'
df = pd.read_csv(csv_file_path)
# 将日期列转换为 datetime 类型
df['日期'] = pd.to_datetime(df['日期'])
# 确保数据按日期升序排列
df = df.sort_values('日期')

# 创建 Dash 应用
app = dash.Dash(__name__)

# 默认起始日期和结束日期对应的下标
start_idx = 2500
end_idx = start_idx + 30

# 创建 K 线图的函数
def create_candlestick_chart(data):
    fig = go.Figure(data=[go.Candlestick(
        x=data['日期'],
        open=data['开盘'],
        high=data['高'],
        low=data['低'],
        close=data['收盘'],
        increasing_line_color='red',  # 红色代表上涨
        decreasing_line_color='green',  # 绿色代表下跌
        # 修改悬浮框显示内容和格式
        hovertext=data.apply(lambda row: (
            f"日期: {row['日期']}<br>"
            f"开盘价: {row['开盘']}<br>"
            f"收盘价: {row['收盘']}<br>"
            f"最高价: {row['高']}<br>"
            f"最低价: {row['低']}<br>"
            f"涨跌金额: {(row['收盘'] - row['开盘']):.2f}<br>"
            f"涨跌幅: {row['涨跌幅']}%"
        ), axis=1),
        hoverinfo="text"
    )])

    fig.update_layout(
        title='K线图',
        xaxis_title='日期',
        yaxis_title='价格',
        xaxis_rangeslider_visible=False,  # 不显示范围滑块
        hovermode='closest',  # 鼠标悬停时显示信息
        height=800  # 增加图表高度（单位：像素）
    )

    return fig

# 应用布局
app.layout = html.Div([
    dcc.Graph(id='k-line-chart', figure=create_candlestick_chart(df.iloc[start_idx:start_idx+1])),  # 初始显示第一天数据
    html.Button("下一天", id="next-day-button", n_clicks=0),
    html.Div(id="date-range")
])

# 回调函数：每次点击按钮，增加一天数据
@app.callback(
    Output('k-line-chart', 'figure'),
    Output('date-range', 'children'),
    Input('next-day-button', 'n_clicks')
)
def update_graph(n_clicks):
    current_end_idx = end_idx + n_clicks + 1

    # 使用下标从 df 中取数据
    data_to_show = df.iloc[start_idx:current_end_idx]

    # 更新 K 线图
    figure = create_candlestick_chart(data_to_show)

    # 显示当前展示的日期范围
    date_range_display = f"展示的日期范围: {data_to_show['日期'].iloc[0].strftime('%Y-%m-%d')} - {data_to_show['日期'].iloc[-1].strftime('%Y-%m-%d')}"

    return figure, date_range_display

# 运行 Dash 应用
if __name__ == '__main__':
    app.run(debug=True)