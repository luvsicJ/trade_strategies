# 现在我有一个csv文件/Users/apple/Desktop/MyProject/TradeStrategies_py/trade_strategies/历史数据/001219-青岛食品-历史数据20150101～20250509.csv
# 里面收录了一只股票的历史交易数据，内容格式如下(排在前面的是最新的交易日，遍历csv文件需要倒序，且日期不完全连续，因为仅包含交易日起)
# "日期","收盘","开盘","高","低","交易量","涨跌幅"
# "2025-5-9","13.75","13.80","13.89","13.65","3.21M","-0.07%"
# "2025-5-8","13.76","13.69","13.77","13.55","3.04M","0.00%"
# "2025-5-7","13.76","13.68","13.88","13.63","4.21M","0.88%"
#
# 我有一个想法，我认为当日股票跌幅达到N时，第二天股价的波动价格中的最高值很大概率会大于昨日的收盘价
# 现在我想统计这个概率，具体方法如下
# 给定一个日期区间，例如[2025-04-02,2025-05-09]
# 倒序遍历csv文件(最前面的是最新的交易记录)，到起始区间2025-04-02时，若昨日2025-04-01的跌幅<=-3% 认为是大跌
# 计算当日2025-04-02最高价 最低价基于前一天2025-04-01收盘价的涨跌幅，依次遍历完整个日期区间，统计出大跌的总天数，以及涨跌幅的区间概率
#
# 然后绘制次日最高价涨幅的分布直方图，横坐标为涨跌幅的区间，[-∞,-3%,-2.5%,...，2.5%,3%,+∞]间隔为N 可修改(默认0.5%)
# 比如2025-04-02基于2025-04-01收盘价的涨跌幅是-0.4%～0.6%，横跨了三个区间[-0.5%,0%],[0%,0.5%],[0.5%,1%]，这三个区间的统计天数都要+1,以此类推
# 使用python实现该程序，生成两个文件，位置为当前文件所在位置的/output文件夹
#   一个是直方图的html文件（histogram.html），横坐标是涨跌幅区间，纵坐标是天数，鼠标放上去的时候要展示落在这个区间的日期列表，图表标题要展示这个日期区间内大跌的总天数
#   一个是k线图(绿色为跌，红色为涨，仅需要绘制指定日期区间部分)的html文件（candlestick.html），标记出大跌后的那个交易日（在k线图上标注昨日跌幅，基于昨日收盘价的(最大跌幅,最大涨幅)，不需要交易量
# 还需要帮我模拟一下 假设我每次都在大跌日买入100股，在次日以高于大跌日收盘价N%的价格卖出100股挂单卖出,每次买入卖出的手续费为5元
# 每笔交易需要生成日期：yyyy-mm-dd日(涨跌幅:{n}%)买入100股@{a}元/股，在yyyy-mm-dd日尝试以{a*(1+N)}元/股卖出，成交结果{成功/失败}，{detail}
# 1.若开盘价高于我挂单的价格，则自动以开盘价成交({detail}内容为：以开盘价@{b}元/股成交，+{(b-a)*100}元)
# 2.若挂单价格<=大跌次日股价的最大值，({detail}内容为：以挂单价@{a*(1+N)}元/股成交，+{N*100}元)
# 3.若挂单价格>大跌次日股价的最大值，则本次交易失败,({detail}内容为：无法以挂单价@{a*(1+N)}元/股成交，当日股票最高价为@{c}元/股，尝试在以后卖出{sell_detail})
# 3.1若本次交易失败，则尝试在大跌次日到指定日期区间最后一天以每日收盘价尝试卖出（当日收盘价>买入时的股价）若到最后都无法卖出，{sell_detail}为：无法卖出，{a}元/股成本价过高，目前亏损：{基于指定日期最后一天的股价算出亏损}；若成功卖出，则{sell_detail}为：在yyyy-mm-dd日以@{d}元/股卖出，+{d-a}*100元
# 最终根据每笔交易的盈亏算出总盈亏，根据未完成的交易统计出当前持有股数，以及每股的平均成本
#可修改的遍历放在全局变量中 方便我修改  中文回答问题，代码的注释和图表的内容说明都要用中文， 文件生成后自动用浏览器打开

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
from datetime import timedelta
import webbrowser

# 全局可修改参数
START_DATE = '2024-01-01'  # 日期区间起始
END_DATE = '2024-12-31'    # 日期区间结束
FALL_THRESHOLD = -0.015     # 大跌阈值（%）
SELL_PROFIT_RATE = 0.02    # 卖出目标盈利比例（%）
HISTOGRAM_INTERVAL = 0.01 # 直方图涨跌幅区间间隔（%）
TRANSACTION_FEE = 5.0      # 每笔交易手续费（元）
SHARES_PER_TRADE = 500     # 每次交易股数

# 确保输出目录存在
OUTPUT_DIR = './output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def read_stock_data(file_path):
    """读取CSV文件并按日期倒序排序"""
    df = pd.read_csv(file_path, encoding='utf-8')
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values('日期', ascending=False).reset_index(drop=True)
    return df

def calculate_big_fall_stats(df, start_date, end_date):
    """统计大跌后次日价格波动概率"""
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df = df[(df['日期'] >= start_date) & (df['日期'] <= end_date)].copy()

    big_fall_days = 0
    histogram_data = {}
    date_lists = {}  # 存储每个区间的日期列表

    for i in range(len(df) - 1):
        today = df.iloc[i]
        yesterday = df.iloc[i + 1]

        # 检查昨日是否大跌
        if yesterday['涨跌幅'].strip('%').replace(',', '') != '':
            yesterday_change = float(yesterday['涨跌幅'].strip('%').replace(',', '')) / 100
            if yesterday_change <= FALL_THRESHOLD:
                big_fall_days += 1

                # 计算次日基于昨日收盘价的最大涨幅和最大跌幅
                yesterday_close = yesterday['收盘']
                today_high_change = (today['高'] - yesterday_close) / yesterday_close
                today_low_change = (today['低'] - yesterday_close) / yesterday_close

                # 确定影响的涨跌幅区间
                min_change = min(today_low_change, today_high_change)
                max_change = max(today_low_change, today_high_change)

                # 计算覆盖的区间
                start_bin = int(min_change / HISTOGRAM_INTERVAL) * HISTOGRAM_INTERVAL
                end_bin = int(max_change / HISTOGRAM_INTERVAL) * HISTOGRAM_INTERVAL + HISTOGRAM_INTERVAL

                for bin_start in np.arange(start_bin, end_bin, HISTOGRAM_INTERVAL):
                    bin_key = f"{bin_start*100:.1f}%~{(bin_start+HISTOGRAM_INTERVAL)*100:.1f}%"
                    histogram_data[bin_key] = histogram_data.get(bin_key, 0) + 1
                    if bin_key not in date_lists:
                        date_lists[bin_key] = []
                    date_lists[bin_key].append(today['日期'].strftime('%Y-%m-%d'))

    return big_fall_days, histogram_data, date_lists

def plot_histogram(big_fall_days, histogram_data, date_lists):
    """绘制次日最高价涨幅分布直方图"""
    x_labels = sorted(histogram_data.keys(), key=lambda x: float(x.split('~')[0].strip('%')))
    y_values = [histogram_data.get(label, 0) for label in x_labels]

    # 创建hover text，包含日期列表
    hover_texts = [
        f"涨跌幅区间: {label}<br>天数: {histogram_data.get(label, 0)}<br>日期: {', '.join(date_lists.get(label, []))}"
        for label in x_labels
    ]

    fig = go.Figure(data=[
        go.Bar(
            x=x_labels,
            y=y_values,
            text=y_values,
            textposition='auto',
            hovertext=hover_texts,
            hoverinfo='text'
        )
    ])

    fig.update_layout(
        title=f'大跌（≤{FALL_THRESHOLD*100:.1f}%）后次日涨跌幅分布（共{big_fall_days}天）',
        xaxis_title='涨跌幅区间',
        yaxis_title='天数',
        bargap=0.1
    )

    output_path = os.path.join(OUTPUT_DIR, 'histogram.html')
    fig.write_html(output_path, full_html=True)
    return output_path

def plot_candlestick(df, start_date, end_date, big_fall_days_info):
    """绘制K线图并标记大跌后交易日"""
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df = df[(df['日期'] >= start_date) & (df['日期'] <= end_date)].copy()

    fig = go.Figure(data=[
        go.Candlestick(
            x=df['日期'],
            open=df['开盘'],
            high=df['高'],
            low=df['低'],
            close=df['收盘'],
            increasing_line_color='red',
            decreasing_line_color='green'
        )
    ])

    # 添加大跌后交易日的标注
    annotations = []
    shapes = []
    for today_date, yesterday_change, max_fall, max_rise in big_fall_days_info:
        annotations.append(
            dict(
                x=today_date,
                y=df[df['日期'] == today_date]['高'].values[0],
                xref="x",
                yref="y",
                text=f"昨日跌幅: {yesterday_change*100:.2f}%<br>最大跌幅: {max_fall*100:.2f}%<br>最大涨幅: {max_rise*100:.2f}%",
                showarrow=True,
                arrowhead=2,
                ax=20,
                ay=-30
            )
        )
        shapes.append(
            dict(
                type="rect",
                x0=today_date - timedelta(days=0.4),
                x1=today_date + timedelta(days=0.4),
                y0=df['低'].min(),
                y1=df['高'].max(),
                fillcolor="yellow",
                opacity=0.2,
                line_width=0
            )
        )

    fig.update_layout(
        title=f'股票K线图（{start_date.strftime("%Y-%m-%d")} 至 {end_date.strftime("%Y-%m-%d")}）',
        xaxis_title='日期',
        yaxis_title='价格',
        annotations=annotations,
        shapes=shapes,
        xaxis_rangeslider_visible=False
    )

    output_path = os.path.join(OUTPUT_DIR, 'candlestick.html')
    fig.write_html(output_path, full_html=True)
    return output_path

def simulate_trading(df, start_date, end_date, sell_profit_rate, transaction_fee, shares_per_trade):
    """模拟大跌日买入次日卖出策略"""
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df = df[(df['日期'] >= start_date) & (df['日期'] <= end_date)].copy()

    total_profit = 0
    holding_shares = 0
    total_cost = 0
    trade_logs = []
    big_fall_days_info = []

    for i in range(len(df) - 1):
        today = df.iloc[i]
        yesterday = df.iloc[i + 1]

        # 检查昨日是否大跌
        if yesterday['涨跌幅'].strip('%').replace(',', '') != '':
            yesterday_change = float(yesterday['涨跌幅'].strip('%').replace(',', '')) / 100
            if yesterday_change <= FALL_THRESHOLD:
                buy_price = yesterday['收盘']
                sell_target_price = buy_price * (1 + sell_profit_rate)

                # 计算次日最大涨跌幅（用于K线图标注）
                max_fall = (today['低'] - buy_price) / buy_price
                max_rise = (today['高'] - buy_price) / buy_price
                big_fall_days_info.append((today['日期'], yesterday_change, max_fall, max_rise))

                # 交易逻辑
                log = f"{yesterday['日期'].strftime('%Y-%m-%d')}日(涨跌幅:{yesterday_change*100:.2f}%)买入{shares_per_trade}股@{buy_price:.2f}，"
                log += f"在{today['日期'].strftime('%Y-%m-%d')}日尝试以@{sell_target_price:.2f}卖出，"

                # 情况1：开盘价高于挂单价
                if today['开盘'] >= sell_target_price:
                    profit = (today['开盘'] - buy_price) * shares_per_trade - 2 * transaction_fee
                    total_profit += profit
                    log += f"成交结果: 成功，以开盘价@{today['开盘']:.2f}成交，+{profit:.2f}元"

                # 情况2：挂单价 <= 当日最高价
                elif sell_target_price <= today['高']:
                    profit = (sell_target_price - buy_price) * shares_per_trade - 2 * transaction_fee
                    total_profit += profit
                    log += f"成交结果: 成功，以挂单价@{sell_target_price:.2f}成交，+{profit:.2f}元"

                # 情况3：挂单价 > 当日最高价，尝试后续卖出
                else:
                    holding_shares += shares_per_trade
                    total_cost += buy_price * shares_per_trade
                    sell_detail = ""

                    # 尝试在后续日期卖出
                    for j in range(i - 1, -1, -1):
                        future_day = df.iloc[j]
                        if future_day['收盘'] > buy_price:
                            profit = (future_day['收盘'] - buy_price) * shares_per_trade - 2 * transaction_fee
                            total_profit += profit
                            holding_shares -= shares_per_trade
                            total_cost -= buy_price * shares_per_trade
                            sell_detail = f"在{future_day['日期'].strftime('%Y-%m-%d')}日以@{future_day['收盘']:.2f}卖出，+{profit:.2f}元"
                            break
                    else:
                        # 若无法卖出，计算截至最后一天的亏损
                        last_price = df.iloc[0]['收盘']
                        loss = (last_price - buy_price) * shares_per_trade
                        sell_detail = f"无法卖出，@{buy_price:.2f}成本价过高，目前股价@{last_price} 这笔交易亏损：{loss:.2f}元"

                    log += f"成交结果: 失败，无法以挂单价@{sell_target_price:.2f}成交，当日股票最高价为@{today['高']:.2f}\n>>>>>尝试在以后卖出:{sell_detail}"

                trade_logs.append(log)

    # 计算平均持股成本
    avg_cost = total_cost / holding_shares if holding_shares > 0 else 0

    return trade_logs, total_profit, holding_shares, avg_cost, big_fall_days_info

def main():
    # 读取数据
    file_path = '/Users/apple/Desktop/MyProject/TradeStrategies_py/trade_strategies/历史数据/002261-拓维信息-历史数据20100101～20250509.csv'
    df = read_stock_data(file_path)

    # 统计大跌后次日波动概率
    big_fall_days, histogram_data, date_lists = calculate_big_fall_stats(df, START_DATE, END_DATE)

    # 绘制直方图
    histogram_path = plot_histogram(big_fall_days, histogram_data, date_lists)

    # 模拟交易
    trade_logs, total_profit, holding_shares, avg_cost, big_fall_days_info = simulate_trading(
        df, START_DATE, END_DATE, SELL_PROFIT_RATE, TRANSACTION_FEE, SHARES_PER_TRADE
    )

    # 绘制K线图
    candlestick_path = plot_candlestick(df, START_DATE, END_DATE, big_fall_days_info)

    # 输出交易结果
    print(f"\n交易模拟结果：")
    for log in trade_logs:
        print(log)
    print(f"\n总盈亏：{total_profit:.2f}元")
    print(f"当前持有股数：{holding_shares}股")
    print(f"每股平均成本：{avg_cost:.2f}元" if holding_shares > 0 else "每股平均成本：0.00元")

    # 自动打开生成的HTML文件
    webbrowser.open('file://' + os.path.abspath(histogram_path))
    webbrowser.open('file://' + os.path.abspath(candlestick_path))

if __name__ == '__main__':
    main()