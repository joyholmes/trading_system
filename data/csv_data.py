from .base import DataSource
import pandas as pd

class CSVDataSource(DataSource):
    def __init__(self, config):
        super().__init__(config)
        self.data_path = config['data_path']
        
    def get_historical_data(self, symbol, start_time, end_time, interval):
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        mask = (df.index >= start_time) & (df.index <= end_time)
        return df.loc[mask]
    
    def get_latest_data(self, symbol):
        df = pd.read_csv(self.data_path)
        return df.iloc[-1:] 