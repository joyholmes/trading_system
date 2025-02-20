from .base import DataSource
import tushare as ts
import pandas as pd
from typing import Dict, Any

class TuShareData(DataSource):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.token = config['token']
        ts.set_token(self.token)
        self.pro = ts.pro_api()
        
    def get_historical_data(
        self, 
        symbol: str, 
        start_time: str, 
        end_time: str,
        interval: str
    ) -> pd.DataFrame:
        # 获取日线数据
        df = self.pro.daily(
            ts_code=symbol,
            start_date=start_time.split(' ')[0].replace('-', ''),
            end_date=end_time.split(' ')[0].replace('-', '')
        )
        
        # 重命名列以统一格式
        df = df.rename(columns={
            'trade_date': 'timestamp',
            'open': 'open',
            'close': 'close',
            'high': 'high',
            'low': 'low',
            'vol': 'volume'
        })
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df.sort_index()
    
    def get_latest_data(self, symbol: str) -> pd.DataFrame:
        df = ts.get_realtime_quotes(symbol)
        # 处理数据格式...
        return df 