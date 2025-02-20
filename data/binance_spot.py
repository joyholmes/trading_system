from .base import DataSource
import pandas as pd
from binance.client import Client
from typing import Dict, Any

class BinanceSpotData(DataSource):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')
        self.client = Client(self.api_key, self.api_secret)
        
    def get_historical_data(
        self, 
        symbol: str, 
        start_time: str, 
        end_time: str,
        interval: str
    ) -> pd.DataFrame:
        # 获取K线数据
        klines = self.client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_time,
            end_str=end_time
        )
        
        # 转换为DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 
            'volume', 'close_time', 'quote_volume', 'trades',
            'buy_base_volume', 'buy_quote_volume', 'ignore'
        ])
        
        # 数据处理
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # 转换数据类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        return df
    
    def get_latest_data(self, symbol: str) -> pd.DataFrame:
        # 获取最新的K线数据
        kline = self.client.get_klines(
            symbol=symbol,
            interval=Client.KLINE_INTERVAL_1MINUTE,
            limit=1
        )
        return self._format_kline_data(kline) 