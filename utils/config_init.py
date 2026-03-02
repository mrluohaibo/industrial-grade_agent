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

    def get_properties(self, prop_name, default=None):
        '''
        Get configuration property with optional default value.

        :param prop_name: a.b.c
        :param default: default value if property not found
        :return: property value or default
        '''
        split_prop = prop_name.split('.')
        temp_conf = self.config
        for key in split_prop:
            temp_conf = temp_conf.get(key)
            if temp_conf is None:
                return default
        return temp_conf if temp_conf is not None else default



application_conf_path  = os.path.join(root_path,"config/application.yaml")
application_conf = Config(application_conf_path)
# print(application_conf.get_properties("mysql.host"))