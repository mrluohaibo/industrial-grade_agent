import utils.logger_config as logger

from bz_core.stock_info_api import StockInfo

if __name__ == '__main__':
    stock_info = StockInfo()
    # stock_info.update_stock_industry()
    # stock_info.read_stock_k_daily_data("601166",'1990-12-19',"2026-01-08")
    # stock_info.query_history_key_data("688125")
    # stock_info.query_all_stock_history_k_daily_data()
    stock_info.query_stock_intro('601166')
    logger.info("-------------ok!!!-------------------------")
