import os.path

import yaml

from bz_core.Constant import root_path


class Config:
    def __init__(self,path):
        self.path = path
        with open(path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)


    def refresh(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def get_properties(self,prop_name):
        '''

        :param prop_name:  a.b.c
        :return:
        '''
        split_prop = prop_name.split('.')
        temp_conf = self.config
        for key in split_prop:
            temp_conf = temp_conf.get(key)
        # print(temp_conf)
        return temp_conf



application_conf_path  = os.path.join(root_path,"config/application.yaml")
application_conf = Config(application_conf_path)
# print(application_conf.get_properties("mysql.host"))