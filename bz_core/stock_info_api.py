import concurrent
import json
import os
import time
from pathlib import PurePath

import requests

import my_akshare as ak
import akshare as remote_ak
import pandas as pd

from bz_core import Constant
from bz_core.Constant import NewsType
from bz_core.stock_dict import stock_dict_zh_2_en, stock_tick_dict, stock_intro_dict
from bz_core.thread_pool_define import handle_daily_stock_data_pool
from utils.logger_config import logger
from utils.StringUtil import StringUtil
from utils.datetime_util import DateTimeUtil
from utils.db_tool_init import mongo_client, td_engine_client
import baostock as bs

class StockInfo():

    def __init__(self):
        # 所有股票信息
        self.stock_db_name = "all_stock_basic"
        # 所有股票日K线数据
        self.stock_k_daily_history_db_name = "stock_daily_k_history"

        self.stock_business_intro = "stock_business_intro"
        # 显示登陆返回信息
        self.bs_login = bs.login()

    def get_all_stock_info(self):
        '''
        获取实时全股票数据
        :return:
        '''
        timestamp_before = time.time() * 1000
        stock_zh_a_spot_em_df = ak.stock_zh_a_spot_em()
        logger.info(f" get stock all info spend {time.time() * 1000 - timestamp_before} ms")
        date_str = DateTimeUtil.now_time_yyyymmdd()
        excel_save_file = self.join_path(Constant.root_path, f"temp_file_save/last_stock_info_{date_str}.xlsx")
        excel_save_file = str(PurePath(excel_save_file).as_posix())
        stock_zh_a_spot_em_df.to_excel(excel_save_file, index=False)
        timestamp_before = time.time() * 1000
        self.handle_stock_pd_data(stock_zh_a_spot_em_df)
        logger.info(f" save stock all info to db spend {time.time() * 1000 - timestamp_before} ms")
        # print(stock_zh_a_spot_em_df)

    def handle_stock_pd_data(self, df):
        heads = [
            # "序号",
            "代码",
            "名称",
            "最新价",
            "涨跌幅",
            "涨跌额",
            "成交量",
            "成交额",
            "振幅",
            "最高",
            "最低",
            "今开",
            "昨收",
            "量比",
            "换手率",
            "市盈率-动态",
            "市净率",
            "总市值",
            "流通市值",
            "涨速",
            "5分钟涨跌",
            "60日涨跌幅",
            "年初至今涨跌幅",
        ]
        all_data = []
        for index, row in df.iterrows():
            # print(f"行索引: {index}")
            # print(f"代码: {row['stock_code']}, 名称: {row['stock_name']}")
            item_row = {}
            has_data = False
            for head_name in heads:
                head_name_en = self.get_en_code_of_zh(head_name)
                if not StringUtil.is_empty(head_name_en):
                    item_row[head_name_en] = row[head_name]
                    if not has_data:
                        has_data = True
            if has_data:
                all_data.append(item_row)
        logger.info(f" refresh all stock info:{len(all_data)}")
        self.insert_stock_info_2_mongo(all_data)

    def insert_stock_info_2_mongo(self, all_data):
        '''
        每天盘后将最新数据更新到全局股票表中
        :param all_data:
        :return:
        '''
        for per_row in all_data:
            unique_stock_code = per_row["stock_code"]
            query = {"stock_code": unique_stock_code}
            match_doc = mongo_client.find_one(self.stock_db_name, query=query)
            if match_doc is not None:
                per_row["update_time"] = DateTimeUtil.now_time_yyyy_mm_dd_hh_mm_ss()
                update_num = mongo_client.update_one(self.stock_db_name, query=query, update={"$set": per_row})
                logger.info(f"stock {unique_stock_code} update {update_num}")
            else:
                per_row["create_time"] = DateTimeUtil.now_time_yyyy_mm_dd_hh_mm_ss()
                mongo_client.insert_one(self.stock_db_name, per_row)

        logger.info(f" insert {self.stock_db_name} info:{len(all_data)}")

    def get_en_code_of_zh(self, zh_name):
        head_name_en = stock_dict_zh_2_en.get(zh_name, "")
        if head_name_en == "":
            logger.error(f"字段 {zh_name} 没有找到对应的en_code")
        return head_name_en

    def parse_xlsx_to_pd(self):
        date_str = DateTimeUtil.now_time_yyyymmdd()
        excel_save_file = self.join_path(Constant.root_path, f"temp_file_save/last_stock_info_{date_str}.xlsx")
        excel_save_file = str(PurePath(excel_save_file).as_posix())
        df = pd.read_excel(excel_save_file)
        self.handle_stock_pd_data(df)

    def join_path(self, a, b):
        return os.path.join(a, b)

    def update_stock_industry(self):
        '''
         获取每只股票的行业
         并将行业形成字典表进行保存
         低频更新，基本可以一个一天更新一次就行了
        :return:
        '''
        # 分页获取 股票信息 1229 2124 更新的id
        last_id = ""
        page_size = 1000
        match_doc = mongo_client.get_cursor_paginated_data(self.stock_db_name, query={}, last_id=last_id,
                                                           page_size=page_size)
        while len(match_doc) > 0:
            last_id = match_doc[-1]["_id"]
            stock_code_list = [item["stock_code"] for item in match_doc]
            self.update_industry_stock(stock_code_list)
            if len(match_doc) < page_size:
                break
            else:
                match_doc = mongo_client.get_cursor_paginated_data(self.stock_db_name, query={}, last_id=last_id,
                                                                   page_size=page_size)

    def update_industry_stock(self, stock_code_list):
        '''

        :param stock_code_list:
        :return:
        '''
        if len(stock_code_list) > 0:
            for stock_code in stock_code_list:
                stock_individual_info_em_df = ak.stock_individual_info_em(symbol=stock_code)
                if stock_individual_info_em_df is not None:
                    stock_info = {}
                    has_data = False
                    for index, row in stock_individual_info_em_df.iterrows():
                        item_name = row["item"]
                        item_value = row["value"]
                        if item_name in ['行业', '上市时间', '总股本', '流通股']:
                            head_name_en = self.get_en_code_of_zh(item_name)
                            if not StringUtil.is_empty(head_name_en):
                                stock_info[head_name_en] = item_value
                                if not has_data:
                                    has_data = True

                    if has_data:
                        self.update_stock_other_info(stock_code, stock_info)

    def update_stock_other_info(self, stock_code, stock_info):
        query = {"stock_code": stock_code}
        match_doc = mongo_client.find_one(self.stock_db_name, query=query)
        if match_doc is not None:
            stock_info["update_time"] = DateTimeUtil.now_time_yyyy_mm_dd_hh_mm_ss()
            update_num = mongo_client.update_one(self.stock_db_name, query=query, update={"$set": stock_info})
            logger.info(f"stock {stock_code} other info update {update_num}")
        else:
            logger.error(f"stock {stock_code} not exist,is impossible")

    def query_stock_tick_store_db(self, stock_code):
        '''
        历史分笔数据
        每个交易日 16:00 提供当日数据; 如遇到数据缺失, 请使用 ak.stock_zh_a_tick_163() 接口(注意数据会有一定差异)
        :param stock_code:
        :return:
        '''
        if StringUtil.is_empty(stock_code):
            logger.error(f"stock {stock_code} not exist,is impossible")
            raise Exception("股票代码不能为空")
        if not stock_code.startswith("s"):
            if stock_code.startswith("6"):
                stock_code = "sh" + stock_code
            elif stock_code.startswith("920"):
                stock_code = "bj" + stock_code
            else:
                stock_code = "sz" + stock_code

        stock_zh_a_tick_tx_js_df = ak.stock_zh_a_tick_tx_js(symbol=stock_code)
        if stock_zh_a_tick_tx_js_df.empty:
            logger.error(f"stock {stock_code} not exist")
            return
        if len(stock_zh_a_tick_tx_js_df) < 3000:
            fail_res = f"stock {stock_code} tick data len {stock_zh_a_tick_tx_js_df.size} not full"
            logger.error(fail_res)
            # raise Exception(fail_res)

        date_str = DateTimeUtil.now_time_yyyymmdd()
        format_date = DateTimeUtil.now_time_yyyy_mm_dd()
        excel_save_file = self.join_path(Constant.root_path,
                                         f"temp_file_save/last_stock_tick_{stock_code}_{date_str}.xlsx")
        stock_zh_a_tick_tx_js_df.to_excel(excel_save_file, index=False)
        self.store_stock_tick_to_db(stock_code,format_date, stock_zh_a_tick_tx_js_df)

    def read_stock_tick_excel_to_db(self, stock_code):
        if not stock_code.startswith("s"):
            if stock_code.startswith("6"):
                stock_code = "sh" + stock_code
            elif stock_code.startswith("920"):
                stock_code = "bj" + stock_code
            else:
                stock_code = "sz" + stock_code

        date_str = DateTimeUtil.now_time_yyyymmdd()
        format_date = DateTimeUtil.now_time_yyyy_mm_dd()
        excel_save_file = self.join_path(Constant.root_path,
                                         f"temp_file_save/last_stock_tick_{stock_code}_{date_str}.xlsx")

        df = pd.read_excel(excel_save_file)
        if df.empty:
            logger.error(f"stock {stock_code} not exist")
            return
        if len(df) <= 3000:
            logger.error(f"stock {stock_code} data not full")
            raise Exception(f"stock {stock_code} tick data len {df.size} not full")

        self.store_stock_tick_to_db(stock_code, format_date, df)

    def store_stock_tick_to_db(self, stock_code, format_date, stock_zh_a_tick_tx_js_df):
        heads = [
            # "序号",
            "成交时间",
            "成交价格",
            "价格变动",
            "成交量",
            "成交金额",
            "性质",
        ]
        self.delete_point_day_tick_data(stock_code, format_date)


        def get_en_code_of_zh_stock_tick(zh_name):
            head_name_en = stock_tick_dict.get(zh_name, "")
            if head_name_en == "":
                logger.error(f"字段 {zh_name} 没有找到对应的en_code")
            return head_name_en

        def get_nature_type(nature):
            if nature == "买盘":
                return 1
            elif nature == "卖盘":
                return -1
            else:
                return 0

        all_data = []

        for index, row in stock_zh_a_tick_tx_js_df.iterrows():
            # print(f"行索引: {index}")
            # print(f"代码: {row['stock_code']}, 名称: {row['stock_name']}")
            item_row = {}
            has_data = False
            for head_name in heads:
                head_name_en = get_en_code_of_zh_stock_tick(head_name)
                if not StringUtil.is_empty(head_name_en):
                    item_row[head_name_en] = row[head_name]
                    if not has_data:
                        has_data = True
            if has_data:
                item_row["trade_day"] = format_date
                item_row["ts"] = DateTimeUtil.str_to_timestamp_ms(
                    format_date + " " + item_row[get_en_code_of_zh_stock_tick("成交时间")])
                item_row["nature_type"] = get_nature_type(item_row[get_en_code_of_zh_stock_tick("性质")])
                all_data.append(item_row)
        logger.info(f" refresh all stock tick info:{len(all_data)}")
        self.insert_stock_tick_to_tdengine(stock_code,format_date, all_data)

    def insert_stock_tick_to_tdengine(self, stock_code,format_date, all_data):
        real_code = stock_code[2:]
        exchange = stock_code[:2]

        def dick_to_tuple(dict_list):
            keys = ["ts", "trade_day", "transaction_time", "transaction_price", "price_change", "trading_volume",
                    "transaction_volume", "nature_type"]
            result = [tuple(entry[k] for k in keys) for entry in dict_list]
            return result

        sql = f'INSERT INTO {stock_code} USING stock_tick TAGS ("{real_code}", "{exchange}")  VALUES (?,?,?,?,?,?,?,?)'
        batch_count = 50
        batch_list = []
        for item in all_data:

            batch_list.append(item)
            if len(batch_list) == batch_count:
                tuple_list = dick_to_tuple(batch_list)
                td_engine_client.insert_many(sql,data = tuple_list)
                batch_list =[]

        if len(batch_list) > 0:
            tuple_list = dick_to_tuple(batch_list)
            td_engine_client.insert_many(sql, data=tuple_list)
            batch_list = []

        logger.info(f"stock code:{stock_code} to tdengine store date {format_date} tick data ok，total len ：{len(all_data)}")

    def delete_point_day_tick_data(self, stock_code, format_date):
        # 插入单条数据

        td_engine_client.execute(f"""
                      DELETE FROM {stock_code}
                        WHERE ts BETWEEN '{format_date} 00:00:00' AND '{format_date} 23:59:59';
                       """)

    def query_all_stock_history_k_daily_data(self):
        # 分页获取 股票信息 1229 2124 更新的id
        last_id = ""
        page_size = 20
        match_doc = mongo_client.get_cursor_paginated_data(self.stock_db_name, query={}, last_id=last_id,
                                                           page_size=page_size)
        end_date = DateTimeUtil.date_to_yyyy_mm_dd_str(DateTimeUtil.time_add_day(-1))

        while len(match_doc) > 0:
            last_id = match_doc[-1]["_id"]
            stock_code_list = [item["stock_code"] for item in match_doc]
            logger.info(f">>> current batch stock code last id is {last_id} size is {len(stock_code_list)}")
            start_time = time.time()
            if len(stock_code_list) > 0:
                for stock_code in stock_code_list:
                    if not self.check_code_date_exist(stock_code, end_date):
                        logger.info(f"start query {stock_code} history k daily data")
                        self.query_history_key_data(stock_code)
                        logger.info(f"end query {stock_code} history k daily data")

                # 虽然是“谁先完成谁先处理”，但每次循环仍会阻塞等待下一个完成的任务。
                # for future in concurrent.futures.as_completed(future_map.keys()):
                #     try:
                #         # 获取每个任务的结果
                #         result = future.result()
                #         logger.info(f"Task code {future_map[future]} returned {result}")
                #     except Exception as exc:
                #         # 如果任务中发生异常，则在这里捕获并处理
                #         logger.error(f"Task code {future_map[future]}  generated an exception: {exc}")
                logger.info(f"history k daily data batch size {len(stock_code_list)} spend time:{time.time()*1000 - start_time*1000} ms")
                if len(match_doc) < page_size:
                    break
                match_doc = mongo_client.get_cursor_paginated_data(self.stock_db_name, query={}, last_id=last_id,
                                                                   page_size=page_size)


    def query_history_key_data(self,stock_code):
        '''
        查询并保存K线数据，北交所暂时不支持
        :param stock_code:
        :return:
        '''

        logger.info('login respond error_code:' + self.bs_login.error_code)
        logger.info('login respond  error_msg:' + self.bs_login.error_msg)
        query_stock_code = ""
        if not stock_code.startswith("s"):
            if stock_code.startswith("6"):
                query_stock_code = "sh." + stock_code
            elif stock_code.startswith("920"):
                query_stock_code = "bj." + stock_code
                logger.error(f"not support stock_code:{stock_code} in beijing ")
                return None
            else:
                query_stock_code = "sz." + stock_code


        #### 获取沪深A股历史K线数据 ####
        # 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
        # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
        # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
        start_date = '1990-12-19'
        end_date = DateTimeUtil.date_to_yyyy_mm_dd_str(DateTimeUtil.time_add_day(-1))
        rs = bs.query_history_k_data_plus(query_stock_code,
                                          "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
                                          start_date=start_date, end_date=end_date,
                                          frequency="d", adjustflag="2")
        logger.info('query_history_k_data_plus respond error_code:' + rs.error_code)
        logger.info('query_history_k_data_plus respond  error_msg:' + rs.error_msg)

        #### 打印结果集 ####
        data_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())

        if rs.error_code == '0':
            result = pd.DataFrame(data_list, columns=rs.fields)
            csv_save_file = self.join_path(Constant.root_path,
                                             f"temp_file_save/history_stock_k_daily_data_{stock_code}_{start_date}_{end_date}.csv")
            csv_save_file = str(PurePath(csv_save_file).as_posix())
            #### 结果集输出到excel文件 ####
            result.to_csv(csv_save_file, index=False)
            logger.info(f"query stock_code {stock_code} startdate {start_date} enddate {end_date} query data {len(result)}")
            self.store_stock_k_daily_data(stock_code, start_date, end_date,result)
            return True
        else:
            logger.error(f"⚠ ⚠ ⚠ ⚠ ⚠ ⚠ error_code {rs.error_code} ,error_msg :{rs.error_msg}")
            return False

    def read_stock_k_daily_data(self, stock_code, start_date, end_date):

        # start_date = '1990-12-19'
        # end_date = '2026-01-08'
        csv_save_file = self.join_path(Constant.root_path,
                                       f"temp_file_save/history_stock_k_daily_data_{stock_code}_{start_date}_{end_date}.csv")

        csv_save_file = str(PurePath(csv_save_file).as_posix())
        df = pd.read_csv(csv_save_file)
        self.store_stock_k_daily_data(stock_code, start_date, end_date,df)


    def store_stock_k_daily_data(self, stock_code, start_date, end_date, result_df):
        '''
         保存数据到mongo
        :param stock_code:
        :param start_date:
        :param end_date:
        :param result:
        :return:
        '''
        all_data = []

        for index, row in result_df.iterrows():
            # print(f"行索引: {index}")
            # print(f"代码: {row['stock_code']}, 名称: {row['stock_name']}")
            item_row = row.to_dict()
            item_row['stock_code'] = stock_code
            all_data.append(item_row)

        self.save_history_k_daily_data_to_mongo(stock_code, all_data)

    def save_history_k_daily_data_to_mongo(self, stock_code, all_data):
        for per_data in all_data:
            query = {
                "stock_code": stock_code,
                "date": per_data['date'],
            }

            update_num = mongo_client.update_one(self.stock_k_daily_history_db_name, query=query, update={"$set": per_data},upsert=True)
            logger.info(f"stock k daily {stock_code},date {per_data["date"]} other info update {update_num}")

    def check_code_date_exist(self, stock_code, end_date):
        query = {
            "stock_code": stock_code,
            "date": end_date,
        }
        doc = mongo_client.find_one(self.stock_k_daily_history_db_name, query=query)
        if doc:
            return True
        else:
            return False

    def query_stock_intro(self,stock_code):
        '''
         主营介绍
        :param stock_code:
        :return:
        '''
        stock_zyjs_ths_df = ak.stock_zyjs_ths(symbol=stock_code)

        def get_en_code_of_zh_stock_intro_dict(zh_name):
            head_name_en = stock_intro_dict.get(zh_name, "")
            if head_name_en == "":
                logger.error(f"字段 {zh_name} 没有找到对应的en_code")
            return head_name_en


        if len(stock_zyjs_ths_df) > 0:
            heads = stock_zyjs_ths_df.columns
            for index, row in stock_zyjs_ths_df.iterrows():
                # print(f"行索引: {index}")
                # print(f"代码: {row['stock_code']}, 名称: {row['stock_name']}")
                item_row = {}
                has_data = False
                for head_name in heads:
                    head_name_en = get_en_code_of_zh_stock_intro_dict(head_name)
                    if not StringUtil.is_empty(head_name_en):
                        item_row[head_name_en] = row[head_name]
                        if not has_data:
                            item_row["stock_code"] = stock_code
                            has_data = True
                if has_data:
                    self.save_stock_buz_intro(stock_code, item_row)

    def save_stock_buz_intro(self, stock_code, item_row):

        query = {
            "stock_code": stock_code,
        }
        update_num = mongo_client.update_one(self.stock_business_intro, query=query, update={"$set": item_row}, upsert=True)
        logger.info(f"stock buz intro  {stock_code} other info update {update_num}")







if __name__ == "__main__":
    stock_info = StockInfo()
    # stock_info.update_stock_industry()
    # stock_info.read_stock_k_daily_data("601166",'1990-12-19',"2026-01-08")
    # stock_info.query_history_key_data("688125")
    # stock_info.query_all_stock_history_k_daily_data()
    # stock_info.query_stock_intro('601166')
    # stock_info.query_news_and_save('601166')
    logger.info("-------------ok!!!-------------------------")
