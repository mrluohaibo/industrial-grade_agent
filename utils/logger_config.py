import logging
import logging.handlers
import os
import colorlog
from bz_core.Constant import root_path


log_colors_config = {
    # 终端输出日志颜色配置
    'DEBUG': 'white',
    'INFO': 'cyan',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bold_red',
}

default_formats = {
    # 终端输出格式
    'color_format': '%(log_color)s%(asctime)s-%(name)s-%(filename)s-:%(lineno)d-%(levelname)s-: %(message)s',
    # 日志输出格式
    'log_format': '%(asctime)s-%(name)s-%(filename)s-:%(lineno)d-%(levelname)s-: %(message)s'
}


def setup_logger(name, log_file, level=logging.INFO):
    # 创建日志目录
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 创建 formatter
    formatter = logging.Formatter(
        default_formats["log_format"]
    )

    # 文件 handler（带日志轮转）
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_file, when='midnight', interval=1, backupCount=30
    )
    file_handler.setFormatter(formatter)

    # 控制台 handler（可选）
    console_handler = logging.StreamHandler()
    formatter = colorlog.ColoredFormatter(default_formats["color_format"], log_colors=log_colors_config)
    console_handler.setFormatter(formatter)

    # 创建 logger
    logger = logging.getLogger(name)
    # 清除已有 handler（防止重复）
    if logger.handlers:
        logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str = None):
    """
    Get or create a logger with the given name.

    Args:
        name: Logger name. If None, returns the default 'stock_info_app' logger.

    Returns:
        Logger instance
    """
    if name is None:
        return logger
    return setup_logger(name, os.path.join(root_path, 'logs/app.log'), level=logging.INFO)


# 确保日志目录存在
log_dir = os.path.join(root_path, 'logs')  # log_path为存放日志的路径
os.makedirs(log_dir, exist_ok=True)
# 使用
logger = setup_logger('stock_info_app', os.path.join(root_path,'logs/app.log'), level=logging.INFO)
logger.info("日志服务启动成功")