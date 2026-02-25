# 协程 基于事件循环，适合高并发异步 I/O,仅仅单线程吧，不适合并行情况

import asyncio
import aiohttp
import time
# import nest_asyncio
import queue
from multiprocessing import cpu_count
from threading import Thread
from utils.logger_config import logger



def my_callback(future):
    result = future.result()
    time_i = int(time.time())
    print(f'{time_i} 返回值: ', result)

def full_task_policy():
    print("队列满了哈")




class AsyncPool(object):
    """
    1. 支持动态添加任务
    2. 支持自动停止事件循环
    3. 支持最大协程数
    """

    def __init__(self, loop=None,semaphore_num=cpu_count(), maxsize=10000,task_queue_policy = None ):
        """
        初始化
        :param loop:
        :param maxsize: 默认0，最大队列等待数量
        :param task_queue_policy: 回调函数,用于弹出
        """
        # 在jupyter需要这个，不然asyncio运行出错
        # nest_asyncio.apply()

        self.semaphore = asyncio.Semaphore(semaphore_num)
        # 获取一个事件循环
        if not loop:
            self.loop = asyncio.new_event_loop()

        self.maxsize = maxsize

        # 队列，先进先出，根据队列是否为空判断，退出协程 任务队列
        self.task_queue = queue.Queue(maxsize=maxsize)

        self.loop_thread = None
        self.loop_running = True
        if self.loop:
            self.start_thread_loop()

        self.full_task_queue_policy = task_queue_policy


    # 添加任务
    def add_task(self,coroutine, callback = my_callback):

        if self.task_queue.qsize() >= self.maxsize:
            print("task queue has full for size {}".format(self.maxsize))
            if self.full_task_queue_policy:
                # 队列满了触发回调,看是否有策略
                self.full_task_queue_policy()
        self.task_queue.put((coroutine, callback))


    # 与callback函数 互斥
    def done(self, fn):
        """
        任务完成
        回调函数
        :param fn:
        :return:
        """
        if fn:
            pass
        # self.q.get()
        # self.q.task_done()

    def wait(self):
        """
        等待任务执行完毕 ,相当于直到队列中的任务清空
        :return:
        """
        self.task_queue.join()




    def _start_thread_loop(self):
        """
        运行事件循环
        :param loop: loop以参数的形式传递进来运行
        :return:
        """
        # 将当前上下文的事件循环设置为循环。
        asyncio.set_event_loop(self.loop)
        # 开始事件循环 ,相当于一直循环
        asyncio.run_coroutine_threadsafe(self.loop_event(), self.loop)
        self.loop.run_forever()

    # 用一个线程包装协程，是否可行，理论上没问题，至少实现了异步,不会阻塞,现在的问题是线程没有执行
    def start_thread_loop(self):
        """
        运行事件循环
        :return:
        """
        # self.loop_thread = Thread(target=self._start_thread_loop, args=(self.loop,))
        self.loop_thread = Thread(target=self._start_thread_loop)

        # 运行线程，同时协程事件循环也会运行
        self.loop_thread.start()

    # 相当于每次都会阻塞一秒
    def stop_thread_loop(self, loop_time=1):
        """
        队列为空，则关闭线程
        :param loop_time:
        :return:
        """

        async def _close_thread_loop():
            """
            关闭线程
            :return:
            """
            while True:
                # 相当于任务不完成，这里就会一直循环
                if self.task_queue.empty():
                    self.loop_running = False
                    self.loop.stop()
                    break
                await asyncio.sleep(loop_time)

        # 等待关闭线程
        asyncio.run_coroutine_threadsafe(_close_thread_loop(), self.loop)

    # 提交 方法和
    async def loop_event(self):

        """
        提交任务到事件循环
        :param func: 异步函数对象
        :param callback: 回调函数
        :return:
        """
        while self.loop_running:

            await self.semaphore.acquire()
            func, callback = self.task_queue.get()
            # 将协程注册一个到运行在线程中的循环，thread_loop 会获得一个环任务
            # 注意：run_coroutine_threadsafe 这个方法只能用在运行在线程中的循环事件使用

            future = asyncio.run_coroutine_threadsafe(func, self.loop)

            # 回调函数封装
            def callback_done(_future):
                try:
                    if callback:
                        callback(_future)
                finally:
                    # 不管错误与否这个都需要被执行
                    self.done(_future)
                    self.semaphore.release()

            # 添加回调函数
            future.add_done_callback(callback_done)


    def release(self, loop_time=1):
        """
        释放线程
        :param loop_time:
        :return:
        """
        self.stop_thread_loop(loop_time)

    def running(self):
        """
        获取当前线程数
        :return:
        """
        return self.task_queue.qsize()


async def thread_example(i):
    # url = "http://httpbin.org/ip"
    # async with aiohttp.ClientSession() as session:
    #     async with session.get(url) as res:
    #         # print(res.status)
    #         # print(res.content)
    #         return await res.text()
    await asyncio.sleep(10)
    return str(i) + ' end'




def create_pool(maxsize=10000):
    pool = AsyncPool(semaphore_num=2 * cpu_count(), maxsize=maxsize, task_queue_policy=full_task_policy)
    logger.info(f"create AsyncPool and queue size {maxsize}")
    return pool

def main():
    # 任务组， 最大协程数
    pool = AsyncPool(semaphore_num = 2*cpu_count(),maxsize=10000,task_queue_policy = full_task_policy)

    # 插入任务任务
    for i in range(20000):
        print(f"put {i}")
        pool.add_task(coroutine = thread_example(i), callback=my_callback)

    # 停止事件循环 关闭协程池的时候使用
    # pool.release()

    # 等待
    # pool.wait()
    print("等待子线程结束...")
    time.sleep(5000)

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("run time: ", end_time - start_time)