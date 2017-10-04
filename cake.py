'''
7. Temperature Tracker
'''

class TempTracker:
    def __init__(self):
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0.0
        self.count = 0
        self.mean = 0
        self.freqs = [0] * 111
        self.mode = None

    def insert(self, x):
        self.max = max(self.max, x)
        self.min = min(self.min, x)
        self.sum += x
        self.count += 1
        self.mean = self.sum / self.count
        self.freqs[x] += 1
        self.update_mode(x)

    def get_max(self):
        return self.max

    def get_min(self):
        return self.min

    def get_mean(self):
        return self.mean

    def get_mode(self):
        return self.mode

    def update_mode(self, x):
        if not self.mode:
            self.mode = x
            return
        if self.freqs[x] > self.freqs[self.mode]:
            self.mode = x
