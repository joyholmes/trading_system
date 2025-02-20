from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Dict, Any

class DataSource(ABC):
    """数据源基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    def get_historical_data(
        self, 
        symbol: str, 
        start_time: str, 
        end_time: str,
        interval: str
    ) -> pd.DataFrame:
        """获取历史数据"""
        pass
    
    @abstractmethod
    def get_latest_data(self, symbol: str) -> pd.DataFrame:
        """获取最新数据"""
        pass 