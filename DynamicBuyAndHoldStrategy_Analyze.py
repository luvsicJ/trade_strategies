import pandas as pd
import numpy as np
import plotly.graph_objects as go
import webbrowser
import itertools
import os
from trade_strategies import DynamicBuyAndHoldStrategy  # 假设 DynamicBuyAndHoldStrategy 已定义

# 参数范围
INITIAL_POSITION_FRACTIONS = [round(0.3 + i * 0.1, 1) for i in range(6)]  # 初始仓位比例列表 [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]，表示首次买入占总资金的比例
PROPORTIONALITY_CONSTANTS = [i for i in range(6, 10)]  # 补仓比例常数 k 列表 [1, 2, 3, 4, 5]，决定补仓金额 M=k*N 中 k 的大小
PRICE_DROP_THRESHOLDS = [[0.01 * i for i in [5, 6, 7, 8, 9, 10, 11, 12, 13]]]  # 价格下跌阈值列表 [[0.02, 0.03, 0.09, 0.12]]，表示触发补仓的下跌百分比 [2%, 3%, 9%, 12%]

INITIAL_CAPITAL = 30000  # 初始资金（人民币），用于策略的起始投资金额
SHARE_LOT = 100  # 每次交易股数的最小单位（整手），需为 100 的倍数，符合A股交易规则
FEE_PER_10000 = 1  # 每10000元交易额的手续费（人民币），用于计算买入和卖出的交易成本
START_DATE = '2023-02-01'  # 数据分析的起始日期，格式为 'YYYY-MM-DD'
END_DATE = '2023-09-01'  # 数据分析的结束日期，格式为 'YYYY-MM-DD'
FILE_PATH = '/Users/apple/Desktop/MyProject/TradeStrategies_py/trade_strategies/历史数据/001219-青岛食品-历史数据20150101～20250509.csv'  # 历史数据 CSV 文件路径

def generate_trading_signals(df, initial_position_fraction, price_drop_thresholds, proportionality_constant):
    """
    使用 DynamicBuyAndHoldStrategy 生成交易信号并返回最终收益率
    规则：在起始日买入 initial_position_fraction 仓位；当价格下跌达到 price_drop_thresholds 时，补仓 M=k*N
    返回：最终收益率 (%)
    """
    # 实例化 DynamicBuyAndHoldStrategy
    strategy = DynamicBuyAndHoldStrategy(
        initial_position_fraction=initial_position_fraction,
        price_drop_thresholds=price_drop_thresholds,
        proportionality_constant=proportionality_constant,
        fee_per_10000=FEE_PER_10000
    )

    # 执行策略
    result = strategy.execute(df, INITIAL_CAPITAL, SHARE_LOT)

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
        'INITIAL_POSITION_FRACTION': {v: [] for v in INITIAL_POSITION_FRACTIONS},
        'PROPORTIONALITY_CONSTANT': {v: [] for v in PROPORTIONALITY_CONSTANTS},
        'PRICE_DROP_THRESHOLDS': {str(v): [] for v in PRICE_DROP_THRESHOLDS}
    }

    for init_pos, k, thresholds in itertools.product(INITIAL_POSITION_FRACTIONS, PROPORTIONALITY_CONSTANTS, PRICE_DROP_THRESHOLDS):
        print(f"Parameters: init_pos={init_pos:.1f}, k={k}, thresholds={thresholds}")
        final_return = generate_trading_signals(
            df, init_pos, thresholds, k
        )
        results.append((init_pos, k, thresholds, final_return))
        avg_returns['INITIAL_POSITION_FRACTION'][init_pos].append(final_return)
        avg_returns['PROPORTIONALITY_CONSTANT'][k].append(final_return)
        avg_returns['PRICE_DROP_THRESHOLDS'][str(thresholds)].append(final_return)

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
        high=df['高'],
        low=df['低'],
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
    labels = [f"IP:{ip:.1f}, K:{k}, T:{','.join([f'{t*100:.0f}%' for t in thresh])}"
              for ip, k, thresh, _ in results]
    returns = [r for _, _, _, r in results]

    fig1.add_trace(go.Bar(
        x=labels,
        y=returns,
        marker_color='blue',
        name='最终收益率'
    ))

    fig1.update_layout(
        title=f'{stock_code}-{stock_name}（{START_DATE}～{END_DATE}）<br>各参数组合的最终收益率<br>',
        xaxis=dict(title='参数组合 (IP: INITIAL_POSITION_FRACTION, K: PROPORTIONALITY_CONSTANT, T: PRICE_DROP_THRESHOLDS)', tickangle=45),
        yaxis=dict(title='最终收益率 (%)'),
        showlegend=False,
        template='plotly_white'
    )

    # 平均收益率折线图
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=[f"{v:.1f}" for v in INITIAL_POSITION_FRACTIONS],
        y=[avg_returns['INITIAL_POSITION_FRACTION'][v] for v in INITIAL_POSITION_FRACTIONS],
        mode='lines+markers',
        name='INITIAL_POSITION_FRACTION',
        line=dict(color='red')
    ))

    fig2.add_trace(go.Scatter(
        x=[str(v) for v in PROPORTIONALITY_CONSTANTS],
        y=[avg_returns['PROPORTIONALITY_CONSTANT'][v] for v in PROPORTIONALITY_CONSTANTS],
        mode='lines+markers',
        name='PROPORTIONALITY_CONSTANT',
        line=dict(color='green')
    ))

    fig2.add_trace(go.Scatter(
        x=[f"[{','.join([f'{t*100:.0f}%' for t in eval(v)])}]" for v in avg_returns['PRICE_DROP_THRESHOLDS'].keys()],
        y=[avg_returns['PRICE_DROP_THRESHOLDS'][v] for v in avg_returns['PRICE_DROP_THRESHOLDS'].keys()],
        mode='lines+markers',
        name='PRICE_DROP_THRESHOLDS',
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
        for init_pos, k, thresh, ret in results:
            thresholds_str = ','.join([f'{t*100:.0f}%' for t in thresh])
            print(
                f"INITIAL_POSITION_FRACTION: {init_pos:.1f}, PROPORTIONALITY_CONSTANT: {k}, "
                f"PRICE_DROP_THRESHOLDS: [{thresholds_str}], 最终收益率: {ret:.2f}%")

        print("\n各参数平均收益率：")
        for param, values in avg_returns.items():
            print(f"{param}:")
            for value, avg_ret in sorted(values.items()):
                if param == 'PRICE_DROP_THRESHOLDS':
                    value_str = f"[{','.join([f'{t*100:.0f}%' for t in eval(value)])}]"
                else:
                    value_str = f"{value:.1f}" if param == 'INITIAL_POSITION_FRACTION' else str(value)
                print(f"  {value_str}: {avg_ret:.2f}%")

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