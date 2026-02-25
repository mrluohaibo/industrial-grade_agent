from datetime import datetime,timedelta

class DateTimeUtil(object):

    @staticmethod
    def now_time_yyyy_mm_dd_hh_mm_ss():
        now = datetime.now()
        parsed_time = now.strftime( "%Y-%m-%d %H:%M:%S")
        return parsed_time


    @staticmethod
    def now_time_yyyy_mm_dd():
        now = datetime.now()
        parsed_time = now.strftime( "%Y-%m-%d")
        return parsed_time

    @staticmethod
    def now_time_yyyymmdd():
        now = datetime.now()
        parsed_time = now.strftime("%Y%m%d")
        return parsed_time


    @staticmethod
    def str_to_timestamp_ms(time_str):
        # 解析字符串为 datetime 对象（假设为本地时区）
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        # 转换为 UTC 时间戳（秒），再乘以 1000 得到毫秒
        timestamp_ms = int(dt.timestamp() * 1000)
        return timestamp_ms

    @staticmethod
    def time_add_day(diff:int):
        now = datetime.now()
        # 计算前一天
        add_time = now + timedelta(days=diff)

        # 如果只需要日期部分（ 不包含 时间 ）
        add_date = add_time.date()
        return add_date

    @staticmethod
    def date_to_yyyy_mm_dd_str(date_obj):
        parsed_time = date_obj.strftime("%Y-%m-%d")
        return parsed_time

