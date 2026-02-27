import time
import threading

class Snowflake:
    def __init__(self, datacenter_id: int, worker_id: int):
        # 支持的最大值
        self.max_datacenter_id = 31  # 5 位
        self.max_worker_id = 31      # 5 位
        self.sequence_bits = 12      # 序列号 12 位

        # 位移偏移量
        self.worker_id_shift = self.sequence_bits
        self.datacenter_id_shift = self.sequence_bits + 5
        self.timestamp_left_shift = self.sequence_bits + 5 + 5

        # 最大序列号（4095）
        self.max_sequence = -1 ^ (-1 << self.sequence_bits)

        # 初始值
        self.worker_id = worker_id
        self.datacenter_id = datacenter_id
        self.sequence = 0
        self.last_timestamp = -1

        # 校验 ID 范围
        if not (0 <= worker_id <= self.max_worker_id):
            raise ValueError(f"worker_id 必须在 0 ~ {self.max_worker_id}")
        if not (0 <= datacenter_id <= self.max_datacenter_id):
            raise ValueError(f"datacenter_id 必须在 0 ~ {self.max_datacenter_id}")

        self.lock = threading.Lock()

    def _current_millis(self):
        return int(time.time() * 1000)

    def generate_id(self) -> int:
        with self.lock:
            timestamp = self._current_millis()

            if timestamp < self.last_timestamp:
                raise RuntimeError("时钟回拨，拒绝生成 ID")

            if timestamp == self.last_timestamp:
                # 同一毫秒内，序列号自增
                self.sequence = (self.sequence + 1) & self.max_sequence
                if self.sequence == 0:
                    # 序列号用完，等待下一毫秒
                    timestamp = self._wait_next_millis(self.last_timestamp)
            else:
                # 新的一毫秒，序列号重置
                self.sequence = 0

            self.last_timestamp = timestamp

            # 拼接 ID
            return (
                ((timestamp - 1288834974657) << self.timestamp_left_shift) |
                (self.datacenter_id << self.datacenter_id_shift) |
                (self.worker_id << self.worker_id_shift) |
                self.sequence
            )

    def _wait_next_millis(self, last_timestamp):
        timestamp = self._current_millis()
        while timestamp <= last_timestamp:
            timestamp = self._current_millis()
        return timestamp



snowflake = Snowflake(datacenter_id=1, worker_id=1)

# ===== 使用示例 =====
# if __name__ == "__main__":
#     # 初始化：数据中心 ID=1，机器 ID=1
#
#     # 生成 ID
#     for _ in range(5):
#         print(snowflake.generate_id())