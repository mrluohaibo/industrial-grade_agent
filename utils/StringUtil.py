


class StringUtil:

    def __init__(self):
        pass

    @staticmethod
    def is_empty(input:str):
        if input is None or input == "" or input.strip() == "":
            return True
        else:
            return False
