import pandas as pd
import numpy as np
import plotly.graph_objects as go
import webbrowser
import os
import math
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

# 全局参数
INITIAL_CAPITAL = 30000  # 初始资金（人民币）
SHARE_LOT = 100  # 每次交易股数为100的倍数
FEE_PER_10000 = 1  # 每10000元交易额收取1元交易费
START_DATE = '2025-04-11'
END_DATE = '2025-05-09'
FILE_PATH = '/Users/apple/Downloads/600993-历史数据.csv'
# 跌买涨卖相关参数
FALL_THRESHOLD = -0.01  # 单日跌幅阈值（-1%）
RISE_THRESHOLD = 0.03  # 卖出上涨阈值（3%）
POSITION_FRACTION = 0.5  # 买入仓位比例（60%）
# 动态补仓相关参数
INITIAL_POSITION_FRACTION = 0.5  # 动态满仓起始买入比例（50%）
PRICE_DROP_THRESHOLDS = [0.02, 0.03, 0.09, 0.12]  # 价格下跌阈值（3%, 6%, 9%, 12%）
PROPORTIONALITY_CONSTANT = 10  # 补仓比例常数 M = k * N

# 策略基类
class TradingStrategy(ABC):
    @abstractmethod
    def get_name(self) -> str:
        """返回策略的中文名称"""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """返回策略的中文描述"""
        pass

    @abstractmethod
    def execute(self, df: pd.DataFrame, initial_capital: float, share_lot: int) -> Dict[str, Any]:
        """
        执行策略
        输入：
            df: 包含['日期', '开盘', '高', '低', '收盘', '涨跌幅']的DataFrame
            initial_capital: 初始资金
            share_lot: 交易手数单位
        输出：
            字典，包含：
                - buy_signals: 买入信号列表 [(日期, 价格, 股数, 盈亏, 手续费), ...]
                - sell_signals: 卖出信号列表 [(日期, 价格, 股数, 盈亏, 手续费, 净交易收益, 对应买入信息), ...]
                - key_points: 关键点列表 [(日期, 事件, 投资组合价值, 盈亏, 手续费, 净交易收益或None, 对应买入信息或None), ...]
                - cumulative_return: 累计收益率列表 [%]
                - daily_positions: 每日持仓列表 [(股数, 股票市值, 闲置资金), ...]
                - log: 操作日志列表 [字符串]
        """
        pass

# 跌买涨卖
class ThresholdTradingStrategy(TradingStrategy):
    def __init__(self, fall_threshold: float, rise_threshold: float, position_fraction: float, fee_per_10000: float):
        self.fall_threshold = fall_threshold
        self.rise_threshold = rise_threshold
        self.position_fraction = position_fraction
        self.fee_per_10000 = fee_per_10000

    def get_name(self) -> str:
        return "跌买涨卖"

    def get_description(self) -> str:
        return (
            f"当股票单日跌幅超过{abs(self.fall_threshold) * 100:.0f}%时，买入{self.position_fraction * 100:.0f}%的可用资金仓位；"
            f"当某买入仓位价格上涨{self.rise_threshold * 100:.0f}%时，卖出该部分持仓。"
        )

    def execute(self, df: pd.DataFrame, initial_capital: float, share_lot: int) -> Dict[str, Any]:
        capital = initial_capital
        positions = []  # (买入日期, 买入价格, 股数, 买入手续费)
        buy_signals = []
        sell_signals = []
        cumulative_return = []
        daily_positions = []
        key_points = []
        log = []

        for i in range(len(df)):
            date = df['日期'][i].strftime('%Y-%m-%d')
            close_price = df['收盘'][i]
            change = df['涨跌幅'][i]
            action = '持有'

            # 计算持仓信息
            total_shares = sum(pos[2] for pos in positions)
            stock_value = total_shares * close_price
            portfolio_value = capital + stock_value
            profit = portfolio_value - initial_capital

            # 规则1：单日跌幅超过阈值，买入
            if change <= self.fall_threshold and capital > 0:
                capital_to_use = capital * self.position_fraction
                shares_to_buy = (capital_to_use // (close_price * share_lot)) * share_lot
                if shares_to_buy > 0:
                    transaction_value = shares_to_buy * close_price
                    fee = math.floor(transaction_value / 10000) * self.fee_per_10000
                    capital -= (transaction_value + fee)
                    positions.append((df['日期'][i], close_price, shares_to_buy, fee))
                    action = f"买入 {shares_to_buy} 股，价格 {close_price:.2f}，手续费 {fee:.2f}"
                    buy_signals.append((df['日期'][i], close_price, shares_to_buy, profit, fee))
                    key_points.append((date, '买入', portfolio_value, profit, fee, None, None))

            # 规则2：检查是否达到上涨阈值，卖出
            shares_to_sell = 0
            buy_positions_sold = []
            transaction_profit = 0
            new_positions = []
            for pos in positions:
                buy_date, buy_price, shares, buy_fee = pos
                price_change = (close_price - buy_price) / buy_price
                if price_change >= self.rise_threshold:
                    shares_to_sell += shares
                    days_held = (df['日期'][i] - buy_date).days
                    gross_profit = (close_price - buy_price) * shares
                    position_profit = gross_profit - buy_fee
                    transaction_profit += position_profit
                    buy_positions_sold.append((buy_date, shares, days_held, buy_price, buy_fee, gross_profit))
                else:
                    new_positions.append(pos)
            positions = new_positions

            if shares_to_sell > 0:
                transaction_value = shares_to_sell * close_price
                sell_fee = math.floor(transaction_value / 10000) * self.fee_per_10000
                capital += (transaction_value - sell_fee)
                transaction_profit -= sell_fee
                buy_dates_str = ", ".join(
                    [f"{bd.strftime('%Y-%m-%d')} ({s}股, 持有{dh}天, 买入价{bp:.2f}, 毛收益+{gp:.2f})"
                     for bd, s, dh, bp, bf, gp in buy_positions_sold])
                action = (f"卖出 {shares_to_sell} 股，价格 {close_price:.2f}，手续费 {sell_fee:.2f}，"
                          f"净交易收益 {transaction_profit:.2f}，对应买入：[{buy_dates_str}]")
                sell_signals.append((df['日期'][i], close_price, shares_to_sell, profit, sell_fee, transaction_profit,
                                     buy_positions_sold))
                key_points.append(
                    (date, '卖出', portfolio_value, profit, sell_fee, transaction_profit, buy_positions_sold))

            # 更新持仓信息
            total_shares = sum(pos[2] for pos in positions)
            stock_value = total_shares * close_price
            portfolio_value = capital + stock_value
            cumulative_return.append((portfolio_value / initial_capital - 1) * 100)
            daily_positions.append((total_shares, stock_value, capital))
            log.append(f"日期: {date}, 操作: {action}, 持股数: {total_shares}, 股票市值: {stock_value:.2f}, "
                       f"闲置资金: {capital:.2f}, 累计收益: {profit:.2f}, 累计收益率: {cumulative_return[-1]:.2f}%")

        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'key_points': key_points,
            'cumulative_return': cumulative_return,
            'daily_positions': daily_positions,
            'log': log
        }
# 跌买涨卖(挂单)
class ThresholdTradingStrategyV2(TradingStrategy):
    def __init__(self, fall_threshold: float, rise_threshold: float, position_fraction: float, fee_per_10000: float):
        self.fall_threshold = fall_threshold
        self.rise_threshold = rise_threshold
        self.position_fraction = position_fraction
        self.fee_per_10000 = fee_per_10000

    def get_name(self) -> str:
        return "跌买涨卖(挂单)"

    def get_description(self) -> str:
        return (
            f"当股票单日跌幅达到{abs(self.fall_threshold) * 100:.0f}%时，以该价格挂单买入{self.position_fraction * 100:.0f}%的可用资金仓位；"
            f"当某买入仓位次日起价格上涨{self.rise_threshold * 100:.0f}%时，以该价格挂单卖出该部分持仓（禁止同日买卖）。"
        )

    def execute(self, df: pd.DataFrame, initial_capital: float, share_lot: int) -> Dict[str, Any]:
        capital = initial_capital
        positions = []  # (买入日期, 买入价格, 股数, 买入手续费)
        buy_signals = []
        sell_signals = []
        cumulative_return = []
        daily_positions = []
        key_points = []
        log = []

        for i in range(len(df)):
            date = df['日期'][i].strftime('%Y-%m-%d')
            open_price = df['开盘'][i]
            high_price = df['高'][i]
            low_price = df['低'][i]
            close_price = df['收盘'][i]
            action = '持有'

            # 计算当日的最大跌幅和涨幅（基于开盘价）
            max_fall = (low_price - open_price) / open_price
            max_rise = (high_price - open_price) / open_price

            # 计算持仓信息
            total_shares = sum(pos[2] for pos in positions)
            stock_value = total_shares * close_price
            portfolio_value = capital + stock_value
            profit = portfolio_value - initial_capital

            # 规则1：单日跌幅达到阈值，挂单买入
            buy_trigger_price = open_price * (1 + self.fall_threshold)  # 挂单价格，例如跌 2% 触发
            if low_price <= buy_trigger_price and capital > 0:  # 当日最低价达到或低于挂单价
                buy_price = max(buy_trigger_price, low_price)  # 以挂单价或最低价中较大者成交
                capital_to_use = capital * self.position_fraction
                shares_to_buy = (capital_to_use // (buy_price * share_lot)) * share_lot
                if shares_to_buy > 0:
                    transaction_value = shares_to_buy * buy_price
                    fee = math.floor(transaction_value / 10000) * self.fee_per_10000
                    capital -= (transaction_value + fee)
                    positions.append((df['日期'][i], buy_price, shares_to_buy, fee))
                    action = (f"买入 {shares_to_buy} 股，挂单价格 {buy_trigger_price:.2f}，"
                              f"成交价格 {buy_price:.2f}，手续费 {fee:.2f}")
                    buy_signals.append((df['日期'][i], buy_price, shares_to_buy, profit, fee))
                    key_points.append((date, '买入', portfolio_value, profit, fee, None, None))

            # 规则2：检查每个持仓是否达到上涨阈值，挂单卖出（禁止同日卖出）
            new_positions = []
            sell_actions = []
            for pos in positions:
                buy_date, buy_price, shares, buy_fee = pos
                sell_trigger_price = buy_price * (1 + self.rise_threshold)  # 挂单卖出价格
                if high_price >= sell_trigger_price and df['日期'][i] != buy_date:  # 当日最高价达到或超过挂单价
                    sell_price = min(sell_trigger_price, high_price)  # 挂单价或最高价中较小者
                    deal_way = "成交价格"
                    if sell_price < open_price:  # 挂单如果比开盘还低，则以开盘价直接成交
                        sell_price = open_price
                        deal_way = "开盘价成交"

                    days_held = (df['日期'][i] - buy_date).days
                    gross_profit = (sell_price - buy_price) * shares
                    transaction_value = shares * sell_price
                    sell_fee = math.floor(transaction_value / 10000) * self.fee_per_10000
                    transaction_profit = gross_profit - buy_fee - sell_fee
                    capital += (transaction_value - sell_fee)

                    # 记录卖出操作
                    buy_positions_sold = [(buy_date, shares, days_held, buy_price, buy_fee, gross_profit)]
                    sell_actions.append(
                        f"卖出 {shares} 股，挂单价格 {sell_trigger_price:.2f}，"
                        f"{deal_way} {sell_price:.2f}，手续费 {sell_fee:.2f}，"
                        f"净交易收益 {transaction_profit:.2f}，"
                        f"\n对应买入：{buy_date.strftime('%Y-%m-%d')} ({shares}股, 持有{days_held}天, 买入价{buy_price:.2f}, 毛收益+{gross_profit:.2f})"
                    )
                    sell_signals.append((df['日期'][i], sell_price, shares, profit, sell_fee, transaction_profit,
                                         buy_positions_sold))
                    key_points.append(
                        (date, '卖出', portfolio_value, profit, sell_fee, transaction_profit, buy_positions_sold))
                else:
                    new_positions.append(pos)
            positions = new_positions

            # 更新 action，如果有卖出操作则合并
            if sell_actions:
                action = "; ".join(sell_actions)

            # 更新持仓信息
            total_shares = sum(pos[2] for pos in positions)
            stock_value = total_shares * close_price
            portfolio_value = capital + stock_value
            cumulative_return.append((portfolio_value / initial_capital - 1) * 100)
            daily_positions.append((total_shares, stock_value, capital))
            log.append(
                f"日期: {date}, 操作: {action}, 股价波动({low_price:.2f}~{high_price:.2f})，日涨跌幅({max_fall*100:.2f}%~{max_rise*100:.2f}%)，持股数: {total_shares}, "
                f"股票市值: {stock_value:.2f}, 闲置资金: {capital:.2f}, "
                f"累计收益: {profit:.2f}, 累计收益率: {cumulative_return[-1]:.2f}%, "
            )

        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'key_points': key_points,
            'cumulative_return': cumulative_return,
            'daily_positions': daily_positions,
            'log': log
        }

# 动态补仓
class DynamicBuyAndHoldStrategy(TradingStrategy):
    def __init__(self, initial_position_fraction: float, price_drop_thresholds: List[float],
                 proportionality_constant: float, fee_per_10000: float):
        self.initial_position_fraction = initial_position_fraction
        self.price_drop_thresholds = price_drop_thresholds
        self.proportionality_constant = proportionality_constant
        self.fee_per_10000 = fee_per_10000

    def get_name(self) -> str:
        return "动态补仓"

    def get_description(self) -> str:
        thresholds_str = ', '.join([f'{n * 100:.0f}%' for n in self.price_drop_thresholds])
        return (f"在起始日买入{self.initial_position_fraction * 100:.0f}%仓位；"
                f"当价格下跌达到[{thresholds_str}]时，补仓M=k*N，其中k={self.proportionality_constant}。")

    def execute(self, df: pd.DataFrame, initial_capital: float, share_lot: int) -> Dict[str, Any]:
        capital = initial_capital
        shares = 0
        buy_signals = []
        sell_signals = []
        key_points = []
        cumulative_return = []
        daily_positions = []
        log = []
        initial_price = df['收盘'].iloc[0]
        triggered_thresholds = set()

        # 起始日买入
        capital_to_use = initial_capital * self.initial_position_fraction
        shares_to_buy = (capital_to_use // (initial_price * share_lot)) * share_lot
        if shares_to_buy > 0:
            transaction_value = shares_to_buy * initial_price
            buy_fee = math.floor(transaction_value / 10000) * self.fee_per_10000
            capital -= (transaction_value + buy_fee)
            shares += shares_to_buy
            buy_signals.append((df['日期'][0], initial_price, shares_to_buy, 0, buy_fee))
            key_points.append((df['日期'][0].strftime('%Y-%m-%d'), '买入', transaction_value, 0, buy_fee, None, None))
            log.append(f"日期: {df['日期'][0].strftime('%Y-%m-%d')}, 操作: 买入 {shares_to_buy} 股，"
                       f"价格 {initial_price:.2f}，手续费 {buy_fee:.2f}")

        for i in range(len(df)):
            date = df['日期'][i].strftime('%Y-%m-%d')
            close_price = df['收盘'][i]
            portfolio_value = shares * close_price + capital
            profit = portfolio_value - initial_capital
            return_pct = (portfolio_value / initial_capital - 1) * 100
            cumulative_return.append(return_pct)
            daily_positions.append((shares, shares * close_price, capital))

            # 计算相对于建仓股价的涨跌幅
            price_change_pct = ((close_price - initial_price) / initial_price) * 100

            # 检查补仓
            price_drop = (initial_price - close_price) / initial_price
            action = '持有'
            for n in self.price_drop_thresholds:
                if price_drop >= n and n not in triggered_thresholds and capital > 0:
                    m = self.proportionality_constant * n
                    capital_to_use = initial_capital * m
                    shares_to_buy = (capital_to_use // (close_price * share_lot)) * share_lot
                    transaction_value = shares_to_buy * close_price
                    buy_fee = math.floor(transaction_value / 10000) * self.fee_per_10000
                    if shares_to_buy > 0:
                        if capital >= (transaction_value + buy_fee):
                            capital -= (transaction_value + buy_fee)
                            shares += shares_to_buy
                            triggered_thresholds.add(n)
                            action = f"补仓 {shares_to_buy} 股，价格 {close_price:.2f}，下跌 {price_drop * 100:.2f}%，补仓比例 {m * 100:.2f}%，手续费 {buy_fee:.2f}"
                            buy_signals.append((df['日期'][i], close_price, shares_to_buy, profit, buy_fee))
                            key_points.append((date, '补仓', portfolio_value, profit, buy_fee, None, None))
                        else:
                            action = f"无法补仓：下跌 {price_drop * 100:.2f}%，需资金 {transaction_value + buy_fee:.2f}，剩余资金 {capital:.2f}"
                    else:
                        action = f"无法补仓：下跌 {price_drop * 100:.2f}%，资金 {capital_to_use:.2f} 不足以购买整手（每手 {close_price * share_lot:.2f}）"
            log.append(f"日期: {date}, 操作: {action}, 收盘价:{close_price}，建仓涨跌幅: {price_change_pct:.2f}%，持股数: {shares}, 股票市值: {shares * close_price:.2f}, "
                       f"闲置资金: {capital:.2f}, 累计收益: {profit:.2f}, 累计收益率: {return_pct:.2f}%")

        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'key_points': key_points,
            'cumulative_return': cumulative_return,
            'daily_positions': daily_positions,
            'log': log
        }


def generate_interactive_html(df, strategies_results, output_path, start_date, end_date, stock_code, stock_name):
    fig = go.Figure()

    # 计算价格范围和偏移量
    price_range = df['高'].max() - df['低'].min()
    offset = price_range * 0.0005  # 0.05% of price range

    # 添加 K 线图
    fig.add_trace(go.Candlestick(
        x=df['日期'],
        open=df['开盘'],
        high=df['高'],
        low=df['低'],
        close=df['收盘'],
        increasing_line_color='red',
        decreasing_line_color='green',
        name='K线'
    ))

    # 添加涨跌幅文本（基于开盘价的最大跌幅和涨幅）
    max_fall = (df['低'] - df['开盘']) / df['开盘']
    max_rise = (df['高'] - df['开盘']) / df['开盘']
    fig.add_trace(go.Scatter(
        x=df['日期'],
        y=df['高'],
        mode='none',
        marker=dict(size=0),
        hovertext=[f"涨跌幅: {mf * 100:.2f}% ~ {mr * 100:.2f}%" for mf, mr in zip(max_fall, max_rise)],
        hoverinfo='text',
        showlegend=False
    ))

    # 颜色和标记配置
    strategy_styles = {
        '跌买涨卖': {
            'buy_color': 'red', 'sell_color': 'green', 'line_color': 'blue',
            'buy_symbol': 'line-ew', 'sell_symbol': 'line-ew'
        },
        '跌买涨卖(挂单)': {
            'buy_color': 'black', 'sell_color': 'blue', 'line_color': 'orange',
            'buy_symbol': 'line-ew', 'sell_symbol': 'circle-dot'
        },
        '动态补仓': {
            'buy_color': 'cyan', 'sell_color': 'magenta', 'line_color': 'pink',
            'buy_symbol': 'line-ew', 'sell_symbol': 'line-ew'
        }
        # 可用的 Plotly 标记符号 (symbol)：
        # 点状：circle, square, diamond, cross, x, pentagon, hexagram, star, hourglass, bowtie
        #      circle-open, square-open, diamond-open, cross-open, x-open, etc. (带 -open 为镂空)
        # 线状：line-ns (垂直线), line-ew (水平线), line-ne (右上斜线), line-nw (左上斜线)
        # 箭头：triangle-up, triangle-down, triangle-left, triangle-right
        #      triangle-ne, triangle-nw, triangle-se, triangle-sw
        #      arrow-up, arrow-down, arrow-left, arrow-right
        # 其他：y-up, y-down, asterisk, hash, cross-thin, x-thin
        # 变体：添加 -open (镂空), -dot (中心点), -open-dot (镂空带中心点)
        # 例如：circle, circle-open, circle-dot, circle-open-dot
    }

    # 为每个策略添加信号和收益率曲线
    title_parts = [f'{stock_code}-{stock_name}（{start_date}～{end_date}）']
    for strategy_name, results in strategies_results.items():
        style = strategy_styles.get(strategy_name, {
            'buy_color': 'black', 'sell_color': 'black', 'line_color': 'black',
            'buy_symbol': 'line-ew', 'sell_symbol': 'line-ew'
        })

        # 调试：打印 sell_signals
        print(f"{strategy_name} sell_signals:", results['sell_signals'])

        # 买入信号
        if results['buy_signals']:
            buy_dates, buy_prices, buy_shares, _, _ = zip(*results['buy_signals'])
            fig.add_trace(go.Scatter(
                x=buy_dates,
                y=[max(p, df['低'].min() * 0.99) for p in buy_prices],  # 限制下限
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=style['buy_color'],
                    symbol=style['buy_symbol'],
                    line=dict(width=2)
                ),
                text=[f'买{s}股 @ {p:.2f}' for s, p in zip(buy_shares, buy_prices)],
                textposition='bottom center',
                hoverinfo='text',
                hovertext=[f'{strategy_name} 买{s}股 @ {p:.2f}' for s, p in zip(buy_shares, buy_prices)],
                name=f'{strategy_name} 买'
            ))

        # 卖出信号（逐个绘制，避免同一天覆盖）
        if results['sell_signals']:
            for idx, (sell_date, sell_price, sell_shares, profit, sell_fee, transaction_profit, buy_positions_sold) in enumerate(results['sell_signals']):
                y_pos = min(sell_price + idx * offset, df['高'].max() * 1.01)  # 限制上限
                y_pos = max(y_pos, df['低'].min() * 0.99)  # 限制下限
                text = (
                    f'卖{sell_shares}股+{transaction_profit:.2f} @{sell_price:.2f}<br>'
                )
                hover_text = (
                    f'{strategy_name} 卖出 {sell_shares}股 @ {sell_price:.2f}<br>'
                    f'净收益: {transaction_profit:.2f}<br>' +
                    '<br>'.join([
                        f'{bd.strftime("%Y-%m-%d")} 买({bs}股, 持{dh}天, 价{bp:.2f}, +{gp:.2f})'
                        for bd, bs, dh, bp, bf, gp in buy_positions_sold
                    ])
                )
                fig.add_trace(go.Scatter(
                    x=[sell_date],
                    y=[y_pos],
                    mode='markers+text',
                    marker=dict(
                        size=15,
                        color=style['sell_color'],
                        symbol=style['sell_symbol'],
                        line=dict(width=2)
                    ),
                    text=[text],
                    textposition='top center',
                    hoverinfo='text',
                    hovertext=[hover_text],
                    name=f'{strategy_name} 卖 #{idx+1}' if idx > 0 else f'{strategy_name} 卖',
                    showlegend=(idx == 0)
                ))

        # 收益率曲线
        fig.add_trace(go.Scatter(
            x=df['日期'],
            y=results['cumulative_return'],
            mode='lines',
            line=dict(color=style['line_color'], width=2),
            name=f'{strategy_name} 收益率 (%)',
            yaxis='y2'
        ))

        # 添加策略描述到标题
        title_parts.append(results['description'])

    # 更新布局
    fig.update_layout(
        title='<br>'.join(title_parts),
        yaxis=dict(
            title='价格 (人民币)',
            side='left',
            showgrid=False,
            range=[df['低'].min() * 0.99, df['高'].max() * 1.01]  # 扩展 y 轴范围
        ),
        yaxis2=dict(
            title='累计收益率 (%)',
            side='right',
            overlaying='y',
            showgrid=False
        ),
        xaxis_rangeslider_visible=True,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )

    # 保存并打开 HTML
    fig.write_html(output_path)
    webbrowser.open(f'file://{os.path.abspath(output_path)}')

# 主函数
def main():
    try:
        if not os.path.exists(FILE_PATH):
            raise FileNotFoundError(f"文件 {FILE_PATH} 不存在，请检查路径或文件名。")

        # 解析文件名
        filename = os.path.basename(FILE_PATH)
        stock_code = filename.split("-")[0]
        stock_name = filename.split("-")[1]

        # 读取和处理数据
        df = pd.read_csv(FILE_PATH)
        df['日期'] = pd.to_datetime(df['日期'])
        df = df[(df['日期'] >= START_DATE) & (df['日期'] <= END_DATE)]
        if df.empty:
            raise ValueError(f"指定日期范围 {START_DATE} 至 {END_DATE} 内没有数据，请检查日期或文件内容。")
        df = df.sort_values('日期').reset_index(drop=True)
        df['涨跌幅'] = df['涨跌幅'].str.rstrip('%').astype(float) / 100

        # 验证必要列
        required_columns = ['日期', '开盘', '高', '低', '收盘', '涨跌幅']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV 文件缺少必要列：{required_columns}")

        # 定义输出目录和文件
        output_dir = os.path.join(os.getcwd(), 'output')
        os.makedirs(output_dir, exist_ok=True)  # 创建 output 目录（如果不存在）
        output_path = os.path.join(output_dir, 'kline_trading_signals.html')

        # 定义策略（在 main() 内部创建，确保类已定义）
        strategies = {
            '跌买涨卖': ThresholdTradingStrategy(FALL_THRESHOLD, RISE_THRESHOLD, POSITION_FRACTION, FEE_PER_10000),
            '跌买涨卖(挂单)': ThresholdTradingStrategyV2(FALL_THRESHOLD, RISE_THRESHOLD, POSITION_FRACTION, FEE_PER_10000),
            '动态补仓': DynamicBuyAndHoldStrategy(INITIAL_POSITION_FRACTION, PRICE_DROP_THRESHOLDS,
                                                  PROPORTIONALITY_CONSTANT, FEE_PER_10000)
        }

        # 配置哪些策略启用
        enabled_strategies = {
            '跌买涨卖(挂单)': False,
            '跌买涨卖': False,
            '动态补仓': True
        }

        # 执行启用的策略
        strategies_results = {}
        for strategy_name, enabled in enabled_strategies.items():
            if enabled:
                strategy = strategies[strategy_name]
                result = strategy.execute(df, INITIAL_CAPITAL, SHARE_LOT)
                result['description'] = strategy.get_description()
                strategies_results[strategy_name] = result

        # 合并输出日志
        print("\n=== 交易总结 ===")
        for strategy_name, result in strategies_results.items():
            print(f"\n{strategy_name}（{result['description']}）：")
            print(f"买入信号: {[(d.strftime('%Y-%m-%d'), p, s, pr, f) for d, p, s, pr, f in result['buy_signals']]}")
            print(f"卖出信号: {[(d.strftime('%Y-%m-%d'), p, s, pr, f, tp, [(bd.strftime('%Y-%m-%d'), bs, dh, bp, bf, gp) for bd, bs, dh, bp, bf, gp in bps]) for d, p, s, pr, f, tp, bps in result['sell_signals']]}")
            print(f"关键事件: {[(d, e, v, p, f, tp, [(bd.strftime('%Y-%m-%d'), bs, dh, bp, bf, gp) for bd, bs, dh, bp, bf, gp in bps] if bps else None) for d, e, v, p, f, tp, bps in result['key_points']]}")
            print(f"最终收益率: {result['cumulative_return'][-1]:.2f}%")
            if not result['buy_signals'] and not result['sell_signals']:
                print("警告：没有生成任何交易信号，请检查数据或调整策略参数。")
            print("\n操作日志：")
            print('\n'.join(result['log']))

        # 生成图表
        generate_interactive_html(
            df, strategies_results, output_path,
            START_DATE, END_DATE, stock_code, stock_name
        )

    except FileNotFoundError as e:
        print(f"错误：{e}")
    except ValueError as e:
        print(f"错误：{e}")
    except Exception as e:
        print(f"未知错误：{e}")

if __name__ == "__main__":
    main()