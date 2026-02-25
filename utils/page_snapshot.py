import os
import unittest

import requests
from lxml import etree
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import time
from PIL import Image
from selenium.webdriver.common.by import By
import io
import numpy as np

from my_akshare.utils.qg_ip_proxy_tool import QingguoProxyIp
from utils.logger_config import logger

from bz_core.Constant import root_path
from utils.random_util import RandomUtil



part_file_dir = os.path.join(root_path, 'temp_part_file')  # 文件保存目录
os.makedirs(part_file_dir, exist_ok=True)

part_file_html_dir = os.path.join(part_file_dir, 'html')  # html文件保存目录
os.makedirs(part_file_html_dir, exist_ok=True)

part_file_pic_dir = os.path.join(part_file_dir, 'pic')  # 图片文件保存目录
os.makedirs(part_file_pic_dir, exist_ok=True)


class ScreenShot:

    def __init__(self,driver_path = "F:/soft/chromedriver/chromedriver.exe",use_proxy: bool = False):
        self.driver_path = driver_path
        self.use_proxy =use_proxy

    def screen_long_pic(self,driver,out_file):
        # 获得页面总高度
        total_height = driver.execute_script("return document.body.parentNode.scrollHeight")

        # 滚动到顶部

        time.sleep(2)
        window_height = driver.get_window_size()['height']  # 窗口高度

        temp_part_file_dir = "temp_part_file/"+out_file
        if not os.path.exists(temp_part_file_dir):
            os.makedirs(temp_part_file_dir)

        last_out_file = f'{out_file}.png'
        if total_height > window_height:
            n = total_height // window_height  # 需要滚动的次数
            driver.execute_script(f"window.scrollTo(0, 0)")
            temp_window_screenshort = rf'{temp_part_file_dir}/screenshort.png'
            driver.save_screenshot(temp_window_screenshort)
            base_mat = np.atleast_2d(Image.open(temp_window_screenshort))

            for i in range(n):
                driver.execute_script(f'document.documentElement.scrollTop={window_height * (i + 1)};')
                time.sleep(2)
                temp_file_path = rf'{temp_part_file_dir}/screenshort_{i}.png'
                driver.save_screenshot(temp_file_path)  # 保存截图
                mat = np.atleast_2d(Image.open(temp_file_path))  # 打开截图并转为二维矩阵
                base_mat = np.append(base_mat, mat, axis=0)  # 拼接图片的二维矩阵

            Image.fromarray(base_mat).save(last_out_file)
        else:
            driver.execute_script(f"window.scrollTo(0, 0)")
            driver.save_screenshot(last_out_file)  # 保存截图

    # 推荐开线程用这个方法全屏截图
    def get_full_image(self,url):


        # chromedriver的路径

        # 设置chrome开启的模式，headless就是无界面模式
        # 一定要使用这个模式，不然截不了全页面，只能截到你电脑的高度
        chrome_options = Options()
        # chrome_options.add_argument('headless')
        chrome_options.add_argument("--headless=new")  # 隐藏窗口
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")


        service = Service(executable_path=self.driver_path)
        driver = Chrome(service = service, options=chrome_options)
        driver.set_window_size(1920, 1080)
        # 控制浏览器写入并转到链接
        driver.get(url)
        time.sleep(3)
        # 接下来是全屏的关键，用js获取页面的宽高，如果有其他需要用js的部分也可以用这个方法
        width = driver.execute_script("return document.documentElement.scrollWidth")
        height = driver.execute_script("return document.documentElement.scrollHeight")
        logger.info(f"page w:{width}, h{height}")
        # 将浏览器的宽高设置成刚刚获取的宽高
        driver.set_window_size(width, height)
        time.sleep(2)
        # 截图并关掉浏览器
        full_screen = os.path.join(root_path,f'temp_part_file/pic/full_screen_{int(time.time()*1000)}_{RandomUtil.random_char(6)}.png')
        driver.save_screenshot(full_screen)
        driver.close()
        return full_screen

    def get_url_html(self,url,use_request: bool=True):
        if use_request:
            headers = {
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
                "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "accept-encoding": "gzip, deflate, br",
                "accept-language": "zh-CN,zh;q=0.9,und;q=0.8,en;q=0.7",
                "cache-control": "no-cache",
                "dnt": "1",
                "pragma": "no-cache",
            }
            if self.use_proxy:
                proxy_ip_port = QingguoProxyIp().random_proxy_ip_port()
            else:
                proxy_ip_port = None

            proxies = None
            if proxy_ip_port is not None:
                proxies = {
                    "http": f"http://{proxy_ip_port}"
                }

            response = requests.get(url,headers=headers,proxies = proxies)
            logger.info("response code is " + str(response.status_code))
            if response.status_code == 200:
                result = response.content.decode(response.apparent_encoding)
                file_current_name = str(int(time.time() * 1000)) + "_" + RandomUtil.random_char(6) + ".html"
                file_path = os.path.join(root_path, "temp_part_file/html/" + file_current_name)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(result)
                return file_path
            else:
                logger.error(f"url {url} response code {response.status_code} then try selenium again")
                return self.get_url_html_from_selenium(url)

        else:
            return self.get_url_html_from_selenium(url)


    def get_url_html_from_selenium(self,url):
        # chromedriver的路径

        # 设置chrome开启的模式，headless就是无界面模式
        # 一定要使用这个模式，不然截不了全页面，只能截到你电脑的高度
        chrome_options = Options()
        # chrome_options.add_argument('headless')
        chrome_options.add_argument("--headless=new")  # 隐藏窗口
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")

        service = Service(executable_path=self.driver_path)
        driver = Chrome(service=service, options=chrome_options)
        driver.set_window_size(1920, 1080)
        # 控制浏览器写入并转到链接
        driver.get(url)
        driver.maximize_window()
        html_content = driver.page_source
        driver.close()
        file_current_name = str(int(time.time() * 1000)) + "_" + RandomUtil.random_char(6) + ".html"
        file_path = os.path.join(root_path, "temp_part_file/html/" + file_current_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return file_path

    def test_method_1(self):
        # executable_path 不是文件路径，是目录路径
        service = Service(executable_path = "F:/soft/chromedriver/chromedriver.exe")
        web = Chrome(service = service)
        # 设置窗口大小为最大
        web.maximize_window()
        web.get('https://www.cnblogs.com/111testing/p/10589338.html')



        time.sleep(5)

        # 截取长图 https://blog.51cto.com/u_16213431/11502888
        self.screen_long_pic(driver=web,out_file='full_screen')

    #
    def test_method_2(self):

        self.get_full_image('https://www.cnblogs.com/111testing/p/10589338.html')








screenShot_tool = ScreenShot()

# screenShot_tool.test_method_2()

# screenShot_tool.get_url_html("https://www.cnblogs.com/111testing/p/10589338.html")