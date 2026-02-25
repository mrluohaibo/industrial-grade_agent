

stock_dict_zh_2_en =  {
    "序号":"index",
    "代码":"stock_code",
    "名称":"stock_name",
    "最新价":"new_quota",
    "涨跌幅":"price_change",
    "涨跌额":"price_change_amount",
    "成交量":"trading_volume",
    "成交额":"transaction_volume",
    "振幅":"amplitude",
    "最高":"highest",
    "最低":"lowest",
    "今开":"today_open",
    "昨收":"yesterday_close",
    "量比":"volume_ratio",
    "换手率":"turnover_rate",
    "市盈率-动态":"pe_dynamic",
    "市净率":"price-to-book_ratio",
    "总市值":"total_market_capitalization",
    "流通市值":"market_capitalization",
    "涨速":"rate_of_increase",
    "5分钟涨跌":"5-minute_price_fluctuation",
    "60日涨跌幅":"60-day_price_change",
    "年初至今涨跌幅":"year-to-date_price_change",
    "行业":"industry",
    "上市时间":"launch_date",
    "总股本":"total_share_capital",
    "流通股":"floating_shares",



}

stock_dict_en_2_zh = {v: k for k, v in stock_dict_zh_2_en.items()}


stock_tick_dict = {
    "成交时间": "transaction_time",
    "成交价格": "transaction_price",
    "价格变动": "price_change",
    "成交金额": "transaction_volume",
    "性质": "nature_type",
    "成交量": "trading_volume",
}

stock_tick_dict_en_2_zh = {v: k for k, v in stock_tick_dict.items()}


stock_intro_dict = {
    "主营业务":"primary_business",
    "产品类型":"product_type",
    "产品名称":"product_name",
    "经营范围":"business_scope",
}

stock_intro_dict_en_2_zh = {v: k for k, v in stock_intro_dict.items()}