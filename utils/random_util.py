import secrets

class RandomUtil:

    @staticmethod
    def random_char(random_len=6):
        clear_chars = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"  # 去掉 0,1,O,l,I 等
        return ''.join(secrets.choice(clear_chars) for _ in range(random_len))