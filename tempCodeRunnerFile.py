from typing import Dict, Any
import yaml
import os
from data.akshare_data import AKShareData
from data.tushare_data import TuShareData
from data.binance_spot import BinanceSpotData
from strategies.ma_cross import MACrossStrategy

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

def main():
    # 加载配置
    config = load_config('config.yaml')
    
    # 初始化数据源
    data_source = get_data_source(config)
    
    # 获取数据
    data = data_source.get_historical_data(
        symbol=config['symbol'],
        start_time=config['start_time'],
        end_time=config['end_time'],
        interval=config['interval']
    )
    
    # 初始化策略
    strategy = MACrossStrategy(config['strategy'])
    
    # 执行回测
    results = strategy.backtest(data)
    
    # 输出结果
    print("\n=== 回测结果 ===")
    print(f"策略收益率: {results['strategy_return']:.2%}")
    print(f"买入持有收益率: {results['buy_hold_return']:.2%}")
    print(f"超额收益率: {results['excess_return']:.2%}")
    print("\n交易记录:")
    for trade in results['trades']:
        print(f"时间: {trade['time']}, 类型: {trade['type']}, "
              f"价格: {trade['price']:.2f}, 数量: {trade['shares']}")
    
    print("\n策略参数:")
    for key, value in results['parameters'].items():
        print(f"{key}: {value}")
    
    print("\n图表已保存为 trading_results.png")

if __name__ == "__main__":
    main() 