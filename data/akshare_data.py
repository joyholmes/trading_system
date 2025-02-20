from .base import DataSource
import akshare as ak
import pandas as pd
from typing import Dict, Any
from datetime import datetime

class AKShareData(DataSource):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.market = config.get('market', 'A股')
        
    def get_historical_data(
        self, 
        symbol: str, 
        start_time: str, 
        end_time: str,
        interval: str
    ) -> pd.DataFrame:
        try:
            # 使用 ak.fund_etf_hist_sina 获取ETF历史数据
            df = ak.fund_etf_hist_sina(
                symbol=symbol
            )
            
            # 打印实际的列名，帮助调试
            print("获取到的数据列名:", df.columns.tolist())
            
            # 根据实际返回的列名重命名
            df = df.rename(columns={
                'date': 'date',
                'open': 'open',
                'close': 'close',
                'high': 'high',
                'low': 'low',
                'volume': 'volume'
            })
            
            # 过滤日期范围
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= start_time) & (df['date'] <= end_time)]
            df.set_index('date', inplace=True)
            
            # 打印处理后的数据，确保格式正确
            print("处理后的数据列名:", df.columns.tolist())
            print("数据前5行:\n", df.head())
            
            return df
            
        except Exception as e:
            print(f"数据获取错误，返回的数据格式为: {df.columns.tolist() if 'df' in locals() else '未获取到数据'}")
            raise Exception(f"获取ETF数据失败: {symbol}, 错误: {str(e)}")
    
    def get_latest_data(self, symbol: str) -> pd.DataFrame:
        try:
            df = ak.fund_etf_spot_em()  # 获取ETF实时行情
            df = df[df['代码'] == symbol]
            return df
        except Exception as e:
            raise Exception(f"获取ETF实时数据失败: {symbol}, 错误: {str(e)}") 