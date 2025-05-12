# 现在我有一个csv文件/Users/apple/Desktop/MyProject/TradeStrategies_py/trade_strategies/历史数据/001219-青岛食品-历史数据20150101～20250509.csv
# 里面收录了一只股票的历史交易数据，内容格式如下(排在前面的是最新的交易日，遍历csv文件需要倒序，且日期不完全连续，因为仅包含交易日)
# "日期","收盘","开盘","高","低","交易量","涨跌幅"
# "2025-5-9","13.75","13.80","13.89","13.65","3.21M","-0.07%"
# "2025-5-8","13.76","13.69","13.77","13.55","3.04M","0.00%"
# "2025-5-7","13.76","13.68","13.88","13.63","4.21M","0.88%"
#
# 我有一个想法，我认为在一个稳定的波动区间，买入当日跌幅较大的股票，在后续日子中有很高概率能涨回来，选择合适的涨幅卖出即可稳定盈利
# 具体操作如下
# 设定预期跌幅D，预期涨幅U， 每次买入股数P，计算股价波动长度L，每次买入卖出手续费X， 账户初始资金为0，每次买入若钱不够则进行充值
# 给定一个日期区间，例如[2025-04-01,2025-05-09]（保证csv数据足够多）
# 倒序遍历 CSV 文件，遍历所有数据，仅对指定日期区间执行交易逻辑（注意计算前一天不要使用日期-1，因为交易日不一定连续）
# 1.先判断当日是否满足某笔交易的卖出条件，读取2025-04-01该日的收盘价b
    # 遍历交易列表中未完成的交易，记当前待交易对象为M
    # 若b >= M.预期卖出价，则更新账户余额，记M.成交结果：完成；M.收益：(b-M.买入价格)*M.买入股数；M.成交日期：yyyy-MM-dd；将该笔交易添加到交易成功列表
    # 若b < M.预期卖出价 且b > M.最高股价，则更新M.最高股价：b，M.最高股价出现日：yyyy-MM-dd
# 2.再判断当日是否满足买入条件，若满足以下条件，将这笔交易添加到交易列表
    # 读取2025-04-01该日的涨跌幅(小数)，收盘价b，计算L日内股价的波动区间[c,d]
    #  若a<D，且b处于[c,d]之间，则查看账户余额是否需要充值，再进行买入，添加到交易列表记M.成交结果：未完成
    # 记录这笔交易的具体信息，买入日期：2025-04-01；当日跌幅：(a*100%)； 买入股数：P；买入价格：b；最近L日内股价波动：[c,d] ；预期卖出价：b*(1+U)
# 3.每天结束最后打印当日的买入卖出情况，和当日回报率（当日证券资产+当日账户闲置资金-目前总共充值的金额）/目前总共充值的金额
# 对输出日志进行优化，保留两位小数；打印每日近L日内的股价波动区间；当日股价的（最低价～最高价）；当日收盘价；
# 对卖出操作，打印对应的买入日期，买入价格，预期卖出价格，持有时间；
# 对于统计数据中的未完成交易，打印买入股价；买入日期；买入股数；预期卖出价；M.最高股价；M.最高股价出现日

# 最终统计数据如下
# 1.交易完成几笔，盈利金额
# 2.交易列表中有几笔数据未完成，用指定日期区间最后一天的股价计算这几笔未完成的交易分别亏损多少钱
# 3.统计证券资产（即持有股数*最后一天股价），账户闲置资金，总共充值的金额，最终盈亏（盈利金额-未完成交易的亏损金额），回报比例（证券资产+账户闲置资金-总共充值的金额）/总共充值的金额


# 使用python实现该程序 中文回答
# 不要这样筛选csv文件，df = df[(df['日期'] >= start_date) & (df['日期'] <= end_date)]，还需要指定日期范围之外的数据来算L日内的股价波动

#默认参数设置如下
# D = -0.02  # 预期跌幅（例如-1%）
# U = 0.02   # 预期涨幅（例如2%）
# P = 100   # 每次买入股数
# L = 20     # 股价波动长度（前20个交易日）
# X = 5      # 每次买入卖出手续费

# 生成k线图html文件，生产规则如下，文件生成后自动用浏览器打开
# 位置为当前文件所在位置的/output文件夹
# 其中标记交易成功的买入点为橙色圆形
# 交易失败的买入点为黑色圆形
# 卖出点为蓝色圆形
# 悬浮框中要有详情日志

import pandas as pd
from datetime import datetime

# 默认参数
D = -0.02  # 预期跌幅
U = 0.02   # 预期涨幅
P = 100    # 每次买入股数
L = 20     # 股价波动长度
X = 5      # 每次买入卖出手续费

# 初始化账户和交易记录
account_balance = 0.0  # 账户余额
total_recharged = 0.0  # 总充值金额
completed_trades = []  # 交易成功列表
pending_trades = []  # 未完成交易列表

# 读取 CSV 文件
csv_file = '/Users/apple/Desktop/MyProject/TradeStrategies_py/trade_strategies/历史数据/001219-青岛食品-历史数据20150101～20250509.csv'
df = pd.read_csv(csv_file)
# 将日期转换为 datetime 格式
df['日期'] = pd.to_datetime(df['日期'])
# 确保数据按日期从旧到新排序（正序遍历）
df = df.sort_values(by='日期', ascending=True).reset_index(drop=True)

# 定义日期区间
start_date = datetime.strptime('2024-08-01', '%Y-%m-%d')
end_date = datetime.strptime('2025-05-09', '%Y-%m-%d')

# 计算前 L 日股价波动区间
def get_price_range(df, current_idx, L):
    # 获取前 L 个交易日的收盘价（当前日期之前）
    if current_idx < L:
        return None  # 数据不足
    prices = df['收盘'].iloc[current_idx - L:current_idx]
    return min(prices), max(prices)

# 主逻辑：正序遍历 CSV 文件
for idx in range(len(df)):
    current_date = df['日期'].iloc[idx]
    # 只处理指定日期区间的数据
    if start_date <= current_date <= end_date:
        close_price = df['收盘'].iloc[idx]
        low_price = df['低'].iloc[idx]
        high_price = df['高'].iloc[idx]
        change_pct = float(df['涨跌幅'].iloc[idx].strip('%')) / 100  # 涨跌幅转为小数
        daily_log = f"日期: {current_date.strftime('%Y-%m-%d')}\n"
        daily_log += f"当日收盘价: {close_price:.2f}\n"
        daily_log += f"当日股价范围: ({low_price:.2f} ~ {high_price:.2f})\n"

        # 获取近 L 日股价波动区间
        price_range = get_price_range(df, idx, L)
        if price_range is None:
            daily_log += "近20日股价波动区间: 数据不足\n"
        else:
            min_price, max_price = price_range
            daily_log += f"近20日股价波动区间: [{min_price:.2f}, {max_price:.2f}]\n"

        # 1. 检查是否满足卖出条件
        sold_trades = []
        for trade in pending_trades[:]:  # 复制列表以避免修改时出错
            expected_sell_price = trade['预期卖出价']
            if close_price >= expected_sell_price:
                # 卖出交易
                profit = (close_price - trade['买入价格']) * trade['买入股数'] - 2 * X
                trade['成交结果'] = '完成'
                trade['收益'] = profit
                trade['成交日期'] = current_date.strftime('%Y-%m-%d')
                completed_trades.append(trade)
                pending_trades.remove(trade)
                sold_trades.append(trade)
                account_balance += close_price * trade['买入股数'] - X
                # 计算持有时间
                buy_date = datetime.strptime(trade['买入日期'], '%Y-%m-%d')
                holding_days = (current_date - buy_date).days
                daily_log += (f"卖出: {trade['买入股数']}股, 价格: {close_price:.2f}, "
                              f"收益: {profit:.2f}\n"
                              f"  买入日期: {trade['买入日期']}, "
                              f"买入价格: {trade['买入价格']:.2f}, "
                              f"预期卖出价: {expected_sell_price:.2f}, "
                              f"持有时间: {holding_days}天\n")
            elif close_price > trade.get('最高股价', 0):
                # 更新最高股价和日期
                trade['最高股价'] = close_price
                trade['最高股价出现日'] = current_date.strftime('%Y-%m-%d')

        # 2. 检查是否满足买入条件
        if price_range is None:
            daily_log += "买入检查: 数据不足，无法计算波动区间\n"
        else:
            min_price, max_price = price_range
            if change_pct < D and min_price <= close_price <= max_price:
                # 需要买入
                cost = close_price * P + X
                if account_balance < cost:
                    recharge = cost - account_balance
                    total_recharged += recharge
                    account_balance += recharge
                    daily_log += f"充值: {recharge:.2f}元\n"
                # 记录交易
                trade = {
                    '买入日期': current_date.strftime('%Y-%m-%d'),
                    '当日跌幅': f"{change_pct * 100:.2f}%",
                    '买入股数': P,
                    '买入价格': close_price,
                    '最近L日内股价波动': [min_price, max_price],
                    '预期卖出价': close_price * (1 + U),
                    '成交结果': '未完成'
                }
                pending_trades.append(trade)
                account_balance -= cost
                daily_log += (f"买入: {P}股, 价格: {close_price:.2f}, "
                              f"预期卖出价: {trade['预期卖出价']:.2f}\n")

        # 3. 计算当日回报率
        securities_value = sum(trade['买入股数'] * close_price for trade in pending_trades)
        total_assets = securities_value + account_balance
        daily_return = (total_assets - total_recharged) / total_recharged if total_recharged > 0 else 0
        daily_log += (f"当日证券资产: {securities_value:.2f}, "
                      f"账户余额: {account_balance:.2f}, "
                      f"回报率: {daily_return * 100:.2f}%\n")

        # 直接打印日志（正序）
        print(daily_log)

# 最终统计
print("\n=== 最终统计 ===")
# 1. 交易完成几笔，盈利金额
total_profit = sum(trade['收益'] for trade in completed_trades)
print(f"交易完成: {len(completed_trades)} 笔, 盈利金额: {total_profit:.2f} 元")

# 2. 未完成交易的亏损
last_day_price = df[df['日期'] <= end_date]['收盘'].iloc[-1]  # 最后一天的收盘价
unrealized_loss = 0
print("未完成交易详情:")
for trade in pending_trades:
    loss = (trade['买入价格'] - last_day_price) * trade['买入股数']
    unrealized_loss += loss
    # 修复：正确格式化最高股价字段
    highest_price = trade.get('最高股价', None)
    highest_price_str = f"{highest_price:.2f}" if highest_price is not None else "无"
    highest_price_date = trade.get('最高股价出现日', "无")
    print(f"  买入日期: {trade['买入日期']}, "
          f"买入股价: {trade['买入价格']:.2f}, "
          f"买入股数: {trade['买入股数']}, "
          f"预期卖出价: {trade['预期卖出价']:.2f}, "
          f"最高股价: {highest_price_str}, "
          f"最高股价出现日: {highest_price_date}")
print(f"未完成交易: {len(pending_trades)} 笔, 未实现亏损: {unrealized_loss:.2f} 元")

# 3. 证券资产、账户余额、总充值、最终盈亏、回报比例
securities_value = sum(trade['买入股数'] * last_day_price for trade in pending_trades)
final_profit_loss = total_profit - unrealized_loss
return_ratio = (securities_value + account_balance - total_recharged) / total_recharged if total_recharged > 0 else 0
print(f"证券资产: {securities_value:.2f} 元")
print(f"账户闲置资金: {account_balance:.2f} 元")
print(f"总共充值金额: {total_recharged:.2f} 元")
print(f"最终盈亏: {final_profit_loss:.2f} 元")
print(f"回报比例: {return_ratio * 100:.2f}%")