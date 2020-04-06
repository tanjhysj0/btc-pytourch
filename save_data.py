#!coding:utf-8
import multiprocessing
import os, time, random
import pandas as pd
import numpy as np
from catalyst import run_algorithm
from catalyst.api import (order_target_percent, symbol)

NAMESPACE = 'ltc_DDPG'
# 开始日期
START_DATE = pd.to_datetime('2019-1-1', utc=True)
# 结速日期
END_DATE = pd.to_datetime('2019-1-2', utc=True)
# 观测历史条数
BAR_COUNT = 9


class LTCENV():
    def run(self):
        run_algorithm(
            capital_base=1000,
            data_frequency="daily",
            initialize=self.__initialize,
            handle_data=self.__handle_data,
            exchange_name='bitfinex',
            algo_namespace=NAMESPACE,
            quote_currency="usd",
            start=START_DATE,
            end=END_DATE
        )

    def __initialize(self, context):
        context.asset = symbol('btc_usd')
        context.set_commission(maker=0.002, taker=0.001)  # 设置手续费

    def __handle_data(self, context, data):
        # 观测历名数据
        short_data = data.history(context.asset,
                                  ('open', 'close', 'high', 'low', 'volume'),
                                  bar_count=int((24*60)/15*100+401)*2,
                                  frequency="15m", )


        short_data.to_csv('5_2w.csv', mode='w', header=False)
        exit()


if __name__ == '__main__':
    LTCENV().run()
