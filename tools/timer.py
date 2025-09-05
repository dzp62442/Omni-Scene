import time
import torch
from collections import OrderedDict

class Timer:
    """ Class to help manage printing simple timing of code execution. """

    def __init__(self, newline=True):
        """
        Args:
            newline (bool): True to print on a new line, False to print on the same line.
        """
        self.newline = newline
        self.times = OrderedDict()
        self.reset()

    def reset(self):
        now = time.time()
        self.start = now
        self.last_time = now
        self.times.clear()

    def add(self, name='default', dt=None):
        if dt is not None:
            self.times[name] = dt

    def update(self, name='default'):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.time()
        dt = now - self.last_time
        self.times[name] = dt
        self.last_time = now

    def summary(self, text='Timer', show=True):
        total = 0.
        for name in self.times:
            total += self.times[name]

        if show:  # 打印到屏幕上
            print('[{}]'.format(text), end=' ')
            for name in self.times:
                dt = self.times[name]
                print('%s=%.4f' % (name, dt), end=' ')
            print('total=%.4f sec {%.2f FPS}' % (total, 1./total), end=' ')
            if self.newline:
                print(flush=True)
            else:
                print(end='\r', flush=True)

        times = self.times.copy()  # 返回一个副本
        self.reset()
        return total, times

# 实例用法
if __name__ == '__main__':
    for i in range(5):
        timer = Timer(newline=True)
        time.sleep(1)
        timer.update('one')
        time.sleep(1)
        timer.update('two')
        total, times = timer.summary(f'Demo{i}', show=True)
        print(times)