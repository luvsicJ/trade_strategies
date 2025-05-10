import pandas as pd
import numpy as np
import plotly.graph_objects as go
import webbrowser
import itertools
import os
import re
from trade_strategies import ThresholdTradingStrategy  # 导入 ThresholdTradingStrategy 类
from trade_strategies import ThresholdTradingStrategyV2

# 参数范围
FALL_THRESHOLDS = [round(-0.01 - i * 0.01, 2) for i in range(7)]  # [-0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07]
RISE_THRESHOLDS = [round(0.02 + i * 0.01, 2) for i in range(7)]  # [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
POSITION_FRACTIONS = [round(0.2 + i * 0.1, 1) for i in range(7)]  # [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# FALL_THRESHOLDS = [-0.01 - i * 0.005 for i in range(9)]  # 从 -0.01 到 -0.05
# RISE_THRESHOLDS = [0.01 + i * 0.005 for i in range(9)]  # 从 0.01 到 0.05
# POSITION_FRACTIONS = [0.4 + i * 0.1 for i in range(3)]

INITIAL_CAPITAL = 20000  # 初始资金（人民币）
SHARE_LOT = 100  # 每次交易股数为100的倍数
FEE_PER_10000 = 1  # 每10000元交易额收取1元手续费


START_DATE = '2025-04-01'
END_DATE = '2025-05-07'
FILE_PATH = '/Users/apple/Desktop/MyProject/TradeStrategies_py/trade_strategies/历史数据/600161-天坛生物-历史数据20100101～20250510.csv'

def generate_trading_signals(df, fall_threshold, rise_threshold, initial_capital, share_lot, position_fraction):
    """
    使用 ThresholdTradingStrategy 生成交易信号并返回最终收益率
    规则：单日跌幅超过fall_threshold时买入position_fraction仓位，某买入点上涨rise_threshold时卖出该部分
    返回：最终收益率 (%)
    """
    # 实例化 ThresholdTradingStrategy
    strategy = ThresholdTradingStrategy(
        fall_threshold=fall_threshold,
        rise_threshold=rise_threshold,
        position_fraction=position_fraction,
        fee_per_10000=FEE_PER_10000
    )

    # 执行策略
    result = strategy.execute(df, initial_capital, share_lot)

    # 提取最终收益率
    final_return = result['cumulative_return'][-1]

    return final_return

def run_parameter_sweep(df):
    """
    遍历参数组合，计算最终收益率
    返回：results (list of tuples), avg_returns (dict)
    """
    results = []
    avg_returns = {
        'FALL_THRESHOLD': {v: [] for v in FALL_THRESHOLDS},
        'RISE_THRESHOLD': {v: [] for v in RISE_THRESHOLDS},
        'POSITION_FRACTION': {v: [] for v in POSITION_FRACTIONS}
    }

    for fall, rise, pos in itertools.product(FALL_THRESHOLDS, RISE_THRESHOLDS, POSITION_FRACTIONS):
        print(f"Parameters: fall={fall:.6f}, rise={rise:.6f}, pos={pos:.6f}")
        final_return = generate_trading_signals(
            df, fall, rise, INITIAL_CAPITAL, SHARE_LOT, pos
        )
        results.append((fall, rise, pos, final_return))
        avg_returns['FALL_THRESHOLD'][fall].append(final_return)
        avg_returns['RISE_THRESHOLD'][rise].append(final_return)
        avg_returns['POSITION_FRACTION'][pos].append(final_return)

    for param in avg_returns:
        for value in avg_returns[param]:
            avg_returns[param][value] = np.mean(avg_returns[param][value]) if avg_returns[param][value] else 0

    return results, avg_returns

def plot_kline_chart(df, stock_code, stock_name, START_DATE, END_DATE):
    """
    绘制K线图并保存为HTML文件到 ./output 目录
    """
    fig = go.Figure(data=[go.Candlestick(
        x=df['日期'],
        open=df['开盘'],
        high=df['高'],  # 修正列名
        low=df['低'],  # 移除多余逗号
        close=df['收盘'],
        name='K线'
    )])

    fig.update_layout(
        title=f'{stock_code}-{stock_name}（{START_DATE}～{END_DATE}）<br>K线图<br>',
        xaxis_title='日期',
        yaxis_title='价格 (人民币)',
        xaxis_rangeslider_visible=False,
        template='plotly_white'
    )

    # 保存到 ./output 目录
    output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'kline_chart.html')
    fig.write_html(output_path)
    webbrowser.open(f'file://{os.path.abspath(output_path)}')

def plot_results(results, avg_returns, stock_code, stock_name, START_DATE, END_DATE):
    """
    绘制收益率图表并保存到 ./output 目录：
    1. 所有参数组合的最终收益率（柱状图）
    2. 每个参数的平均收益率（折线图）
    """
    # 参数组合柱状图
    fig1 = go.Figure()
    labels = [f"F:{f:.4f}, R:{r:.4f}, P:{p:.4f}" for f, r, p, _ in results]
    returns = [r for _, _, _, r in results]

    fig1.add_trace(go.Bar(
        x=labels,
        y=returns,
        marker_color='blue',
        name='最终收益率'
    ))

    fig1.update_layout(
        title=f'{stock_code}-{stock_name}（{START_DATE}～{END_DATE}）<br>各参数组合的最终收益率<br>',
        xaxis=dict(title='参数组合 (F: FALL_THRESHOLD, R: RISE_THRESHOLD, P: POSITION_FRACTION)', tickangle=45),
        yaxis=dict(title='最终收益率 (%)'),
        showlegend=False,
        template='plotly_white'
    )

    # 平均收益率折线图
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=[f"{v:.4f}" for v in FALL_THRESHOLDS],  # 使用字符串格式化显示高精度
        y=[avg_returns['FALL_THRESHOLD'][v] for v in FALL_THRESHOLDS],
        mode='lines+markers',
        name='FALL_THRESHOLD',
        line=dict(color='red')
    ))

    fig2.add_trace(go.Scatter(
        x=[f"{v:.4f}" for v in RISE_THRESHOLDS],  # 使用字符串格式化显示高精度
        y=[avg_returns['RISE_THRESHOLD'][v] for v in RISE_THRESHOLDS],
        mode='lines+markers',
        name='RISE_THRESHOLD',
        line=dict(color='green')
    ))

    fig2.add_trace(go.Scatter(
        x=[f"{v:.4f}" for v in POSITION_FRACTIONS],  # 使用字符串格式化显示高精度
        y=[avg_returns['POSITION_FRACTION'][v] for v in POSITION_FRACTIONS],
        mode='lines+markers',
        name='POSITION_FRACTION',
        line=dict(color='blue')
    ))

    fig2.update_layout(
        title=f'{stock_code}-{stock_name}（{START_DATE}～{END_DATE}）<br>各参数的平均收益率<br>',
        xaxis=dict(title='参数值'),
        yaxis=dict(title='平均收益率 (%)'),
        showlegend=True,
        template='plotly_white'
    )

    # 保存到 ./output 目录
    output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)
    fig1.write_html(os.path.join(output_dir, 'parameter_sweep_results.html'))
    fig2.write_html(os.path.join(output_dir, 'parameter_avg_returns.html'))
    webbrowser.open(f'file://{os.path.abspath(os.path.join(output_dir, "parameter_avg_returns.html"))}')
    webbrowser.open(f'file://{os.path.abspath(os.path.join(output_dir, "parameter_sweep_results.html"))}')

def main():
    try:
        if not os.path.exists(FILE_PATH):
            raise FileNotFoundError(f"文件 {FILE_PATH} 不存在，请检查路径或文件名。")

        filename = os.path.basename(FILE_PATH)
        stock_code = filename.split("-")[0]
        stock_name = filename.split("-")[1]

        df = pd.read_csv(FILE_PATH)
        df['日期'] = pd.to_datetime(df['日期'])

        df = df[(df['日期'] >= START_DATE) & (df['日期'] <= END_DATE)]
        if df.empty:
            raise ValueError(f"指定日期范围 {START_DATE} 至 {END_DATE} 内没有数据，请检查日期或文件内容。")

        df = df.sort_values('日期').reset_index(drop=True)
        df['涨跌幅'] = df['涨跌幅'].str.rstrip('%').astype(float) / 100

        required_columns = ['日期', '开盘', '高', '低', '收盘', '涨跌幅']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV 文件缺少必要列：{required_columns}")

        # 运行参数遍历
        results, avg_returns = run_parameter_sweep(df)

        # 打印结果
        print("\n参数组合最终收益率：")
        for fall, rise, pos, ret in results:
            print(
                f"FALL_THRESHOLD: {fall:.4f}, RISE_THRESHOLD: {rise:.4f}, POSITION_FRACTION: {pos:.4f}, 最终收益率: {ret:.2f}%")

        print("\n各参数平均收益率：")
        for param, values in avg_returns.items():
            print(f"{param}:")
            for value, avg_ret in sorted(values.items()):  # 按值排序以清晰显示
                print(f"  {value:.4f}: {avg_ret:.2f}%")

        # 绘制图表
        plot_results(results, avg_returns, stock_code, stock_name, START_DATE, END_DATE)
        plot_kline_chart(df, stock_code, stock_name, START_DATE, END_DATE)

    except FileNotFoundError as e:
        print(f"错误：{e}")
    except ValueError as e:
        print(f"错误：{e}")
    except Exception as e:
        print(f"未知错误：{e}")

if __name__ == "__main__":
    main()
