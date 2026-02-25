import requests
import json
import traceback
import logging as logger
from utils.config_init import application_conf

# 青果代理
class QingguoProxyIp:
    def __init__(self):
        self.key = application_conf.get_properties("ip_proxy.key")
        self.pwd = application_conf.get_properties("ip_proxy.pwd")


    def random_proxy_ip_port(self):
        # 按量
        key = self.key
        pwd = self.pwd
        # 按时

        get_ip_url = f"https://share.proxy.qg.net/get?key={key}&pwd={pwd}"
        proxy_ip_port = None
        try:
            session = requests.session()
            response = session.get(get_ip_url)
            result = response.content.decode("utf-8")
            json_content = json.loads(result)
            code = json_content["code"]
            if code == 'SUCCESS':
                obj_data = json_content["data"]
                proxy_ip_port = obj_data[0]["server"]
                logger.info("request proxy ip success,res is {}".format(proxy_ip_port))
            else:
                logger.error("not get proxy ip success")

        except Exception as e:
            info = traceback.format_exc()
            print(info)
            print(e)

        return proxy_ip_port