import json
import os.path
import time
from pathlib import Path

import baostock as bs
import requests
from requests import Session

from bz_core import Constant
from utils.StringUtil import StringUtil
from utils.datetime_util import DateTimeUtil
from utils.db_tool_init import mongo_client
from utils.logger_config import logger
from utils.random_util import RandomUtil
from utils.pdf_to_markdown import convert_pdf_to_markdown
from pathlib import PurePath


class StockNewsInfo():

    def __init__(self):
        self.cninfo_stock_basic_info = "cninfo_stock_basic_info"
        self.cninfo_news_primitive_info = "cninfo_news_primitive_info"
        # 显示登陆返回信息
        self.bs_login = bs.login()

    def query_news_and_save(self, symbol):
        '''
        https://www.cninfo.com.cn/
        获取财经新闻 type 新闻 1 公告 2
        :param url:
        :return:
        '''
        # stock_news_em_df = ak.stock_news_em(symbol=symbol)
        # print(stock_news_em_df)
        #
        # stock_news_main_cx_df = ak.stock_news_main_cx()
        # print(stock_news_main_cx_df)
        if StringUtil.is_empty(symbol):
            raise Exception(f"symbol is empty")
        cninfo_stock_basic_info = self.query_cninfo_basic_info(symbol)
        orgId = None
        if cninfo_stock_basic_info:
            orgId = cninfo_stock_basic_info["orgId"]
        if orgId is None:
            raise Exception(f"orgId is empty with code {symbol}")

        end_date = DateTimeUtil.now_time_yyyy_mm_dd()
        session = requests.session()
        session.headers.update({
                                   'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'})
        init_response = session.get("https://www.cninfo.com.cn/")
        if init_response.status_code == 200:
            page_num = 1
            page_size = 30
            # fulltext 公告，relation 调研
            news_type = "fulltext"
            while True:

                params = {
                    'pageNum': str(page_num),
                    'pageSize': str(page_size),
                    'column': 'szse',
                    'tabName': news_type,
                    'plate': '',
                    'stock': f'{symbol},{orgId}',
                    'searchkey': '',
                    'secid': '',
                    'category': '',
                    'trade': '',
                    'seDate': f'2023-01-18~{end_date}',
                    'sortName': '',
                    'sortType': '',
                    'isHLtitle': 'true',
                }

                session.headers.update({'Content-Type': 'application/x-www-form-urlencoded'})
                data_response = session.post("https://www.cninfo.com.cn/new/hisAnnouncement/query", data=params)
                if data_response.status_code == 200:
                    result = json.loads(data_response.text)
                    # logger.info(result)
                    news_list = result["announcements"]
                    if len(news_list) > 0:
                        self.save_news_to_db(news_list)
                    hasMore = result["hasMore"]
                    if hasMore :
                        page_num = page_num + 1
                        time.sleep(1)
                    else:
                        logger.info(f"in page {page_num} break")
                        break

                    if len(news_list) < page_size:
                        logger.info(f"in page {page_num} break")
                        break

                else:
                    logger.error(f"data response error {data_response.status_code},error data {data_response.content}")



    def query_all_stock_baisc_info(self):
        '''
        https://www.cninfo.com.cn/
        获取财经新闻 type 新闻 1 公告 2
        :param url:
        :return:
        '''
        # stock_news_em_df = ak.stock_news_em(symbol=symbol)
        # print(stock_news_em_df)
        #
        # stock_news_main_cx_df = ak.stock_news_main_cx()
        # print(stock_news_main_cx_df)
        session = requests.session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'})
        init_response = session.get("https://www.cninfo.com.cn/")
        if init_response.status_code == 200:


            data_response = session.get("https://www.cninfo.com.cn/new/data/szse_stock.json")
            if data_response.status_code == 200:
                result = json.loads(data_response.text)
                list_stock = result["stockList"]
                if len(list_stock) > 0:
                    self.save_and_update_cninfo_stock_basic(list_stock)

            else:
                logger.error(f"data response error {data_response.status_code},error data {data_response.content}")

    def save_and_update_cninfo_stock_basic(self, list_stock):
        for stock in list_stock:
            query={
                "code": stock["code"],
            }

            mongo_client.update_one(self.cninfo_stock_basic_info, query, update={"$set": stock}, upsert=True)

        logger.info(f"stock cninfo stock basic info update {len(list_stock)}")


    def query_cninfo_basic_info(self,stock_code):
        query = {
            "code": stock_code,
        }

        doc = mongo_client.find_one(self.cninfo_stock_basic_info, query)
        return doc

    def save_news_to_db(self, news_list):
        for item_new in news_list:
            query = {
                "announcementId": item_new["announcementId"],
                "secCode": item_new["secCode"],
            }
            mongo_client.update_one(self.cninfo_news_primitive_info, query, update={"$set": item_new}, upsert=True)

    def query_pdf_and_not_get(self):
        '''
        db.cninfo_news_primitive_info.updateMany(
          {"pdf_to_md_file":null },
          { $set: { "pdf_down_load": "0" } }
        )



        :return:
        '''
        last_id = None
        page_size = 1000

        query = {
           "pdf_down_load": {"$ne": 1},
        }

        match_doc = mongo_client.get_cursor_paginated_data(self.cninfo_news_primitive_info, query=query, last_id=last_id,
                                                           page_size=page_size)
        while len(match_doc) > 0:
            last_id = match_doc[-1]["_id"]
            self.update_pdf_save_path(match_doc)
            if len(match_doc) < page_size:
                break
            else:
                match_doc = mongo_client.get_cursor_paginated_data(self.cninfo_news_primitive_info, query=query, last_id=last_id,
                                                                   page_size=page_size)

    def update_pdf_save_path(self,docList):
        if len(docList) > 0:
            for item_doc in docList:
                pdf_url = item_doc["adjunctUrl"]
                if pdf_url:
                    pdf_url = "https://static.cninfo.com.cn/" + pdf_url
                else:
                    logger.error(f"pdf url error {pdf_url} not exist")
                    continue

                save_pdf_file = self.query_pdf_file(pdf_url)
                if save_pdf_file:
                    md_pdf_file = self.convert_pdf_to_markdown(save_pdf_file)
                    query = {
                        "announcementId": item_doc["announcementId"],
                        "secCode": item_doc["secCode"],
                    }
                    update = {
                        "pdf_down_load":1,
                        "save_pdf_file":save_pdf_file,
                        "pdf_to_md_file":md_pdf_file,
                    }

                    update_num = mongo_client.update_one(self.cninfo_news_primitive_info, query, update={"$set": update}, upsert=True)
                    if update_num > 0:
                        logger.info(f"update one pdf url for code {item_doc["secCode"]} newsid {item_doc["announcementId"]} ,{update_num} success")



    def convert_pdf_to_markdown(self,pdf_local_file_path):
        '''
        pip install pymupdf

        :param pdf_local_file_path:
        :return:
        '''

        md_save_file = os.path.join(Constant.root_path,
                                         f"temp_part_file/md/news_{int(time.time()*1000)}_{RandomUtil.random_char(6)}.md")

        native_md_save_file =  str(PurePath(md_save_file).as_posix())
        logger.info(f"pdf_local_file_path:{pdf_local_file_path} start convert pdf to markdown")
        start_time = time.time()*1000
        convert_pdf_to_markdown(pdf_local_file_path, native_md_save_file)
        logger.info(f"covert pdf {native_md_save_file} to markdown {native_md_save_file} spend time {time.time()*1000 - start_time} ms")
        return native_md_save_file



    def query_pdf_file(self,url):
        '''
         返回
        :param url:
        :return:
        '''


        pdf_save_file = os.path.join(Constant.root_path,
                                         f"temp_part_file/pdf/news_{int(time.time()*1000)}_{RandomUtil.random_char(6)}.pdf")

        native_pdf_save_file = str(PurePath(pdf_save_file).as_posix())
        response = requests.get(url)

        # 检查请求是否成功
        if response.status_code == 200:
            with open(native_pdf_save_file, 'wb') as f:
                f.write(response.content)  # 以二进制模式写入
            logger.info(f"PDF 已成功下载并保存为 {native_pdf_save_file}")
            return native_pdf_save_file
        else:
            logger.error(f"下载失败，状态码: {response.status_code}")
            return None

if __name__ == "__main__":
    stockNewsInfo = StockNewsInfo()
    # stockNewsInfo.query_all_stock_baisc_info()
    # stockNewsInfo.query_news_and_save("601166")
    stockNewsInfo.query_pdf_and_not_get()
    path = "F:/python_pro/tick_info/temp_part_file/pdf/1224938931.pdf"
    # native_md_save_file = stockNewsInfo.convert_pdf_to_markdown(path)
    logger.info("-------------ok!!!-------------------------")
