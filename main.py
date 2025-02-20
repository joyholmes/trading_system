from typing import Dict, Any
import yaml
import os
import json
import csv
from datetime import datetime
from pathlib import Path
from data.akshare_data import AKShareData
from data.tushare_data import TuShareData
from data.binance_spot import BinanceSpotData
from strategies.ma_cross import MACrossStrategy
import pandas as pd

def load_config(config_path: str) -> Dict[str, Any]:
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建配置文件的完整路径
    config_file = os.path.join(current_dir, config_path)
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件未找到: {config_file}")

def get_data_source(config: Dict[str, Any]):
    """根据配置选择数据源"""
    source_type = config['data_source']['type']
    if source_type == 'akshare':
        return AKShareData(config['data_source']['akshare'])
    elif source_type == 'tushare':
        return TuShareData(config['data_source']['tushare'])
    elif source_type == 'binance':
        return BinanceSpotData(config['data_source']['binance'])
    else:
        raise ValueError(f"Unsupported data source type: {source_type}")

def create_results_dir(strategy_name: str) -> str:
    """创建结果目录"""
    # 创建基础结果目录
    base_dir = Path("results")
    base_dir.mkdir(exist_ok=True)
    
    # 创建带时间戳的策略目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = base_dir / f"{timestamp}_{strategy_name}"
    
    # 如果目录已存在，添加随机后缀
    if result_dir.exists():
        import random
        result_dir = base_dir / f"{timestamp}_{strategy_name}_{random.randint(1000, 9999)}"
    
    result_dir.mkdir(exist_ok=True)
    
    # 更新最新结果的链接
    latest_link = base_dir / "latest"
    try:
        # 在 Windows 上，需要先删除已存在的目录或链接
        if latest_link.exists():
            if latest_link.is_dir():
                import shutil
                shutil.rmtree(latest_link)
            else:
                latest_link.unlink()
        
        # 直接复制最新结果到 latest 目录
        import shutil
        shutil.copytree(result_dir, latest_link)
            
    except Exception as e:
        print(f"Warning: 无法创建 latest 目录: {e}")
        print(f"结果将只保存在: {result_dir}")
    
    return str(result_dir)

def save_results(results: Dict[str, Any], result_dir: str, config: Dict[str, Any]) -> None:
    """保存回测结果"""
    # 保存回测结果摘要
    summary = {
        "performance": {
            "strategy_return": results['strategy_return'],
            "trade_count": len(results['trades'])
        },
        "parameters": results['parameters']
    }
    
    with open(f"{result_dir}/results.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    # 保存交易记录
    trades_df = pd.DataFrame(results['trades'])
    trades_df.to_csv(f"{result_dir}/trades.csv", index=False)
    
    # 生成绩效报告
    performance_report = f"""回测结果报告
===========================

1. 回测参数
- 回测区间：{config['start_time']} 至 {config['end_time']}
- 交易标的：{config['symbol']}
- 数据周期：{config['interval']}

2. 策略表现
- 初始资金：{config['strategy']['initial_capital']:,.2f}
- 最终资金：{results['holdings'][-1]['value']:,.2f}
- 总收益率：{results['strategy_return']:.2%}
- 交易次数：{len(results['trades'])}

3. 收益率对比
- 策略收益率：{results['strategy_return']:.2%}
- 定投收益率：{results['dip_return']:.2%}
"""
    # 添加基准收益率对比
    if 'benchmark_returns' in results:
        for name, returns in results['benchmark_returns'].items():
            final_return = returns.iloc[-1]
            performance_report += f"- {name}收益率：{final_return:.2%}\n"
    
    performance_report += f"""
4. 策略参数
- 短期均线：{config['strategy']['short_window']}
- 长期均线：{config['strategy']['long_window']}
- 止损：{config['strategy']['stop_loss']:.2%}
- 止盈：{config['strategy']['take_profit']:.2%}
- 仓位管理：{config['strategy']['position_management']['type']}
"""
    
    with open(f"{result_dir}/performance.txt", 'w', encoding='utf-8') as f:
        f.write(performance_report)

def main():
    # 加载配置
    config = load_config('config.yaml')
    
    # 初始化数据源
    data_source = get_data_source(config)
    
    # 获取主要交易标的数据
    data = data_source.get_historical_data(
        symbol=config['symbol'],
        start_time=config['start_time'],
        end_time=config['end_time'],
        interval=config['interval']
    )
    
    # 获取基准数据
    benchmarks = {}
    # 添加主基准（交易标的）
    benchmarks['标的'] = data.copy()
    
    # 获取市场基准数据
    for bench in config['benchmark']['market']:
        bench_data = data_source.get_historical_data(
            symbol=bench['symbol'],
            start_time=config['start_time'],
            end_time=config['end_time'],
            interval=config['interval']
        )
        benchmarks[bench['name']] = bench_data
    
    # 初始化策略
    strategy = MACrossStrategy(config['strategy'])
    
    # 创建结果目录
    result_dir = create_results_dir("MA_Cross")
    
    # 执行回测
    results = strategy.backtest(data, result_dir, benchmarks)
    
    # 保存结果
    save_results(results, result_dir, config)
    
    # 输出结果位置
    print(f"\n回测结果已保存至: {result_dir}")
    print("包含以下文件：")
    print("- results.json: 回测结果摘要")
    print("- trades.csv: 交易记录明细")
    print("- trading_results.png: 策略表现图表")
    print("- performance.txt: 详细绩效分析")

if __name__ == "__main__":
    main() 