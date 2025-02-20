from .base import Strategy
import pandas as pd
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
from datetime import datetime

class MACrossStrategy(Strategy):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.short_window = config.get('short_window', 5)
        self.long_window = config.get('long_window', 20)
        self.stop_loss = config.get('stop_loss', 0.02)
        self.take_profit = config.get('take_profit', 0.05)
        self.initial_capital = config.get('initial_capital', 100000)
        
        # 基础仓位配置
        base_holding_config = config.get('base_holding', {})
        self.base_holding_enabled = base_holding_config.get('enabled', True)
        self.base_holding_ratio = base_holding_config.get('ratio', 0.2)
        self.dynamic_adjust = base_holding_config.get('dynamic_adjust', True)
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # 计算指标
        df['MA_short'] = df['close'].rolling(window=self.short_window).mean()
        df['MA_long'] = df['close'].rolling(window=self.long_window).mean()
        
        # 计算MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 计算RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 计算趋势强度
        df['trend_strength'] = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()
        
        # 生成信号
        df['signal'] = 0
        
        # 买入条件：需要同时满足多个条件
        buy_condition = (
            (df['MA_short'] > df['MA_long']) &                    # 均线金叉
            (df['MACD'] > df['Signal']) &                        # MACD金叉
            ((df['RSI'] < 40) | (df['trend_strength'] > 1.0))    # RSI低位或强势趋势
        )
        
        # 卖出条件：满足任一条件即卖出
        sell_condition = (
            (df['MA_short'] < df['MA_long']) |                   # 均线死叉
            (df['MACD'] < df['Signal']) |                       # MACD死叉
            (df['RSI'] > 75) |                                  # RSI超买
            (df['trend_strength'] < -1.5)                       # 趋势转弱
        )
        
        # 设置信号
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        return df
    
    def backtest(self, data: pd.DataFrame, result_dir: str, benchmarks: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """执行回测"""
        df = self.generate_signals(data)
        
        # 添加日志：检查数据处理后的状态
        print("\n=== 数据预处理检查 ===")
        print(f"处理后的数据行数: {len(df)}")
        print(f"数据起始日期: {df.index[0]}")
        print(f"数据结束日期: {df.index[-1]}")
        print(f"是否存在 NaN 值: {df.isnull().any().any()}")
        if df.isnull().any().any():
            print("NaN 值所在列：")
            print(df.isnull().sum())
        
        # 初始化结果
        position = 0
        entry_price = 0
        trades = []
        capital = self.initial_capital
        current_value = self.initial_capital
        holdings = []
        
        # 建立基础仓位
        if self.base_holding_enabled:
            initial_price = df['close'].iloc[0]
            base_shares = int((self.initial_capital * self.base_holding_ratio * 0.99) // initial_price)
            if base_shares > 0:
                capital -= base_shares * initial_price
                trades.append({
                    'time': df.index[0],
                    'type': 'buy',
                    'price': initial_price,
                    'shares': base_shares,
                    'note': '建立基础仓位'
                })
                position = 1  # 标记为持仓状态
                entry_price = initial_price
                print(f"\n=== 建立基础仓位 ===")
                print(f"基础仓位股数: {base_shares}")
                print(f"基础仓位金额: {base_shares * initial_price:.2f}")
                print(f"剩余资金: {capital:.2f}")
        
        # 记录初始状态
        current_value = capital
        if position == 1:
            current_value += base_shares * initial_price
        
        holdings.append({
            'time': df.index[0],
            'value': current_value
        })
        
        # 交易循环
        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            current_time = df.index[i]
            
            # 动态调整基础仓位（如果启用）
            if self.base_holding_enabled and self.dynamic_adjust:
                if position == 0 and df['signal'].iloc[i] == 1:
                    # 在做多信号出现时建立或增加基础仓位
                    available_capital = capital * self.base_holding_ratio
                    additional_shares = int((available_capital * 0.99) // current_price)
                    if additional_shares > 0:
                        capital -= additional_shares * current_price
                        trades.append({
                            'time': current_time,
                            'type': 'buy',
                            'price': current_price,
                            'shares': additional_shares,
                            'note': '增加基础仓位'
                        })
                        position = 1
                        entry_price = current_price
                elif position == 1 and df['signal'].iloc[i] == -1:
                    # 在做空信号出现时可以减少基础仓位
                    current_shares = trades[-1]['shares']
                    reduce_shares = int(current_shares * 0.5)  # 最多减少一半基础仓位
                    if reduce_shares > 0:
                        capital += reduce_shares * current_price
                        trades.append({
                            'time': current_time,
                            'type': 'sell',
                            'price': current_price,
                            'shares': reduce_shares,
                            'note': '减少基础仓位'
                        })
            
            if position == 0 and df['signal'].iloc[i] == 1:
                # 开仓
                position = 1
                entry_price = current_price
                
                try:
                    # 计算仓位大小
                    position_size = self.calculate_position_size(df, i)
                    if pd.isna(position_size):  # 添加检查
                        print(f"警告：计算得到无效仓位，使用默认值")
                        position_size = self.config.get('base_position', 0.3)
                    
                    available_capital = capital * position_size
                    shares = int((available_capital * 0.99) // current_price)  # 转换为整数
                    
                    # 添加日志：检查买入计算
                    print(f"\n=== 买入信号检查 ({current_time}) ===")
                    print(f"当前价格: {current_price}")
                    print(f"可用资金: {available_capital}")
                    print(f"计算得到的仓位比例: {position_size}")
                    print(f"计算得到的股数: {shares}")
                    
                    if shares > 0:  # 只在有效股数时执行交易
                        capital -= shares * current_price
                        trades.append({
                            'time': current_time,
                            'type': 'buy',
                            'price': current_price,
                            'shares': shares
                        })
                    else:
                        position = 0  # 如果无法买入，恢复为空仓状态
                        print("警告：计算得到的股数为0，取消交易")
                        
                except Exception as e:
                    print(f"执行买入交易时出错: {e}")
                    position = 0  # 发生错误时恢复为空仓状态
                    continue
                
            elif position == 1:
                # 检查止损止盈
                profit_pct = (current_price - entry_price) / entry_price
                shares = trades[-1]['shares']  # 获取最近一次买入的股数
                
                if profit_pct <= -self.stop_loss or profit_pct >= self.take_profit or df['signal'].iloc[i] == -1:
                    # 平仓
                    position = 0
                    
                    # 添加日志：检查卖出计算
                    print(f"\n=== 卖出信号检查 ({current_time}) ===")
                    print(f"当前价格: {current_price}")
                    print(f"入场价格: {entry_price}")
                    print(f"盈亏比例: {profit_pct:.2%}")
                    print(f"卖出股数: {shares}")
                    
                    capital += shares * current_price
                    trades.append({
                        'time': current_time,
                        'type': 'sell',
                        'price': current_price,
                        'shares': shares
                    })
            
            # 更新每日持仓价值
            current_value = capital
            if position == 1:
                current_value += shares * current_price
            
            # 只有当时间不同时才记录，避免重复
            if not holdings or holdings[-1]['time'] != current_time:
                holdings.append({
                    'time': current_time,
                    'value': current_value
                })
        
        # 确保最后一天的数据被记录
        if holdings[-1]['time'] != df.index[-1]:
            holdings.append({
                'time': df.index[-1],
                'value': current_value
            })
        
        # 计算策略收益率
        strategy_return = ((current_value - self.initial_capital) / self.initial_capital) * 100
        
        # 添加日志：检查最终结果
        print("\n=== 回测结果检查 ===")
        print(f"总交易次数: {len(trades)}")
        print(f"初始资金: {self.initial_capital}")
        print(f"最终资金: {current_value}")
        print(f"策略收益率: {strategy_return:.2f}%")
        
        # 计算基准收益率
        benchmark_returns = {}
        if benchmarks:
            for name, bench_data in benchmarks.items():
                # 确保基准数据与回测期间对齐
                bench_data = bench_data[bench_data.index >= df.index[0]]
                bench_data = bench_data[bench_data.index <= df.index[-1]]
                
                # 计算基准收益率（修正计算方法）
                initial_price = bench_data['close'].iloc[0]
                bench_returns = pd.Series(index=df.index)
                bench_returns = ((bench_data['close'] - initial_price) / initial_price) * 100
                benchmark_returns[name] = bench_returns
        
        # 计算定投收益率
        dip_returns = self._calculate_dip_returns(df)
        final_dip_return = dip_returns.iloc[-1]
        
        # 生成图表
        self._plot_results(df, trades, holdings, benchmark_returns, result_dir)
        
        return {
            'trades': trades,
            'holdings': holdings,
            'strategy_return': strategy_return,
            'benchmark_returns': benchmark_returns,
            'dip_return': final_dip_return,  # 添加定投收益率
            'parameters': {
                'short_window': self.short_window,
                'long_window': self.long_window,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit
            }
        }
    
    def _calculate_dip_returns(self, df: pd.DataFrame, monthly_amount: float = 2000) -> pd.Series:
        """计算定投收益率"""
        # 将日期索引转换为datetime
        dates = pd.to_datetime(df.index)
        
        # 获取每月第一个交易日
        monthly_dates = dates[dates.is_month_start]
        if len(monthly_dates) == 0:  # 如果没有月初日期，取每月第一个交易日
            monthly_dates = dates.groupby([dates.year, dates.month]).first()
        
        # 初始化定投数据
        total_investment = 0
        total_shares = 0
        dip_returns = pd.Series(index=df.index, dtype=float)
        
        # 计算每个交易日的定投收益率
        for current_date in df.index:
            # 如果是月初第一个交易日，执行定投
            if current_date in monthly_dates:
                price = df.loc[current_date, 'close']
                shares = monthly_amount / price
                total_shares += shares
                total_investment += monthly_amount
            
            # 计算当前市值和收益率
            if total_investment > 0:  # 避免除以零
                current_value = total_shares * df.loc[current_date, 'close']
                return_rate = ((current_value - total_investment) / total_investment) * 100  # 修正：使用百分比表示
            else:
                return_rate = 0
            
            dip_returns[current_date] = return_rate
        
        return dip_returns

    def _plot_results(self, df: pd.DataFrame, trades: list, holdings: list, 
                     benchmark_returns: Dict[str, pd.Series], result_dir: str) -> None:
        """绘制交易结果图表"""
        # 设置中文字体
        try:
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 尝试使用微软雅黑
        except:
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 如果没有微软雅黑，使用黑体
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置图表样式
        plt.rcParams.update({
            'figure.figsize': (15, 12),
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.labelsize': 10,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'lines.linewidth': 1.5,
            'lines.markersize': 8,
            'font.size': 10
        })
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[2, 1])
        
        # 1. 上半部分：价格和交易信号
        # 绘制价格
        ax1.plot(df.index, df['close'], label='价格', color='gray', alpha=0.7)
        
        # 绘制移动平均线
        ax1.plot(df.index, df['MA_short'], label=f'{self.short_window}日均线', 
                 color='orange', alpha=0.7)
        ax1.plot(df.index, df['MA_long'], label=f'{self.long_window}日均线', 
                 color='blue', alpha=0.7)
        
        # 标记买卖点
        buy_points = [t for t in trades if t['type'] == 'buy']
        sell_points = [t for t in trades if t['type'] == 'sell']
        
        # 买入点
        if buy_points:
            buy_times = [t['time'] for t in buy_points]
            buy_prices = [t['price'] for t in buy_points]
            ax1.scatter(buy_times, buy_prices, color='red', marker='^', s=100, 
                       label='买入信号', zorder=5)
            
            # 添加买入点注释
            for i, (time, price) in enumerate(zip(buy_times, buy_prices)):
                ax1.annotate(f'B{i+1}\n{price:.2f}',  # 移除 ¥ 符号
                            (time, price), 
                            xytext=(10, 10),
                            textcoords='offset points',
                            fontsize=8,
                            color='red')
        
        # 卖出点
        if sell_points:
            sell_times = [t['time'] for t in sell_points]
            sell_prices = [t['price'] for t in sell_points]
            ax1.scatter(sell_times, sell_prices, color='green', marker='v', s=100, 
                       label='卖出信号', zorder=5)
            
            # 添加卖出点注释
            for i, (time, price) in enumerate(zip(sell_times, sell_prices)):
                ax1.annotate(f'S{i+1}\n{price:.2f}',  # 移除 ¥ 符号
                            (time, price), 
                            xytext=(10, -20),
                            textcoords='offset points',
                            fontsize=8,
                            color='green')
        
        ax1.set_title('价格走势与交易信号', pad=15)
        ax1.set_xlabel('日期')
        ax1.set_ylabel('价格')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # 2. 下半部分：收益率对比
        holdings_df = pd.DataFrame(holdings)
        holdings_df.set_index('time', inplace=True)
        
        # 计算策略收益率序列
        strategy_returns = pd.Series(index=df.index)
        for h in holdings:
            strategy_returns[h['time']] = (h['value'] - self.initial_capital) / self.initial_capital * 100
        strategy_returns = strategy_returns.ffill()
        
        # 计算定投收益率
        dip_returns = self._calculate_dip_returns(df)
        
        # 绘制收益率曲线
        ax2.plot(strategy_returns.index, strategy_returns, 
                 label='策略收益率', color='blue', linewidth=2)
        
        # 绘制定投收益率曲线
        ax2.plot(dip_returns.index, dip_returns,
                 label='定投收益率', 
                 color='purple', 
                 linewidth=2,
                 linestyle='-.')
        
        # 绘制基准收益率曲线
        colors = ['gray', 'green', 'orange']
        for (name, returns), color in zip(benchmark_returns.items(), colors):
            ax2.plot(returns.index, returns, 
                     label=f'{name}收益率', 
                     color=color, 
                     alpha=0.7,
                     linestyle='--')
        
        # 添加零线
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        
        ax2.set_title('收益率对比', pad=15)
        ax2.set_xlabel('日期')
        ax2.set_ylabel('收益率(%)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(f"{result_dir}/trading_results.png", dpi=300, bbox_inches='tight')
        plt.close()

    def calculate_position_size(self, df: pd.DataFrame, current_idx: int) -> float:
        """动态计算仓位大小"""
        try:
            # 计算波动率
            volatility = df['close'].rolling(window=10).std()  # 缩短窗口，提高敏感度
            ma_diff = (df['MA_short'] - df['MA_long']) / df['MA_long']
            
            # 如果数据不足，返回基础仓位
            if pd.isna(volatility.iloc[current_idx]):
                return self.config.get('base_position', 0.3)
            
            # 计算波动率因子（更激进的设置）
            recent_volatility = volatility.iloc[max(0, current_idx-10):current_idx+1]
            if len(recent_volatility) == 0 or recent_volatility.max() == 0:
                vol_factor = 1
            else:
                # 使用相对波动率，增加弹性
                vol_factor = (volatility.iloc[current_idx] / recent_volatility.mean())
                vol_factor = min(3.0, vol_factor)  # 最高3倍杠杆
            
            # 计算趋势强度因子（更敏感的设置）
            trend_factor = abs(ma_diff.iloc[current_idx]) * 5  # 增加趋势响应度
            
            # 计算RSI因子（更激进的仓位调整）
            current_rsi = df['RSI'].iloc[current_idx]
            if current_rsi <= 30:  # 超卖区域
                rsi_factor = 2.0    # 更激进的加仓
            elif current_rsi >= 70:  # 超买区域
                rsi_factor = 0.5    # 更激进的减仓
            else:
                rsi_factor = 1.0 + (50 - current_rsi) / 50  # 线性调整因子
            
            # 计算动量因子
            price_change = (df['close'].iloc[current_idx] / df['close'].iloc[max(0, current_idx-5)] - 1)
            momentum_factor = 1 + abs(price_change) * 3
            
            # 计算最终仓位（更激进的组合）
            base_position = self.config.get('base_position', 0.3)
            position_size = base_position * vol_factor * (1 + trend_factor) * rsi_factor * momentum_factor
            
            # 确保返回有效值（允许更大的仓位）
            max_position = self.config.get('max_position', 0.8)
            position_size = min(max(0.15, position_size), max_position)  # 提高最小仓位
            
            return position_size
            
        except Exception as e:
            print(f"计算仓位时出错: {e}")
            return self.config.get('base_position', 0.3)

    def kelly_position_size(self, win_rate: float, profit_ratio: float, loss_ratio: float) -> float:
        """使用凯利公式计算仓位"""
        kelly_fraction = (win_rate * profit_ratio - (1 - win_rate) * loss_ratio) / profit_ratio
        return max(0, min(kelly_fraction, self.config.get('max_position', 0.8)))

    def execute_batch_order(self, signal: int, current_price: float, capital: float) -> Dict:
        """分批执行订单"""
        batch_sizes = [0.4, 0.3, 0.3]  # 分三次建仓，每次使用的资金比例
        holding_period = [0, 2, 5]      # 各批次的间隔天数
        
        total_shares = 0
        remaining_capital = capital
        
        for size, delay in zip(batch_sizes, holding_period):
            if self.can_execute_batch(delay):  # 检查是否满足执行条件
                batch_capital = capital * size
                shares = (batch_capital * 0.99) // current_price  # 考虑手续费
                total_shares += shares
                remaining_capital -= shares * current_price
        
        return {
            'shares': total_shares,
            'remaining_capital': remaining_capital
        } 