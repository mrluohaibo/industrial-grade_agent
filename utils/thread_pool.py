from concurrent.futures import ThreadPoolExecutor


class CustomThreadPool:

    def __init__(self,max_workers=100,thread_name_prefix = "custom_thread_pool"):
        self.pool  = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=thread_name_prefix)

    def add_task(self,func,*args,**kwargs):

        future = self.pool.submit(func,*args,**kwargs)
        return future

    def shutdown(self,wait=True):
        '''
        "Clean-up the resources associated with the Executor.

        清理与执行器关联的资源

        :return:
        '''
        self.pool.shutdown(wait)



if __name__ == '__main__':
    pass