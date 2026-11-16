import threading
import time
import psutil
import glob
import os


class ResourceMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.cpu = 0
        self.ram_used = 0
        self.ram_total = 0
        self.gpu = "N/A"
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def get_gpu_util(self):
        gpu_dirs = glob.glob("/sys/class/devfreq/*gpu*")
        if not gpu_dirs:
            return "N/A"

        try:
            cur_freq_file = os.path.join(gpu_dirs[0], "cur_freq")
            max_freq_file = os.path.join(gpu_dirs[0], "max_freq")
            with open(cur_freq_file, "r") as f:
                cur = int(f.read().strip())
            with open(max_freq_file, "r") as f:
                mx = int(f.read().strip())
            return f"{(cur / mx) * 100:.0f}%"
        except Exception:
            return "N/A"

    def monitor(self):
        while self.running:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory()
            ram_used = ram.used / (1024 * 1024)
            ram_total = ram.total / (1024 * 1024)
            gpu = self.get_gpu_util()

            with self.lock:
                self.cpu = cpu
                self.ram_used = ram_used
                self.ram_total = ram_total
                self.gpu = gpu

            time.sleep(self.interval)

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.monitor, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def get_stats(self):
        with self.lock:
            return self.cpu, self.ram_used, self.ram_total, self.gpu
