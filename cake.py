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

'''
8. Balanced Binary Tree
'''

class BinaryTreeNode:

    def __init__(self, value):
        self.value = value
        self.left  = None
        self.right = None

    def insert_left(self, value):
        self.left = BinaryTreeNode(value)
        return self.left

    def insert_right(self, value):
        self.right = BinaryTreeNode(value)
        return self.right

class Solution:

    def __init__(self):
        self.min = float('inf')
        self.max = float('-inf')

    def is_superbalanced(root):
        if not root:
            return True
        q = [(root, 0)]
        while q:
            n, d = q.pop()
            if not n:
                self.min = min(self.min, d)
                self.max = max(self.max, d)
                if abs(self.max - self.min) > 1:
                    return False
            else:
                q.append((root.left, d+1))
                q.append((root.right, d+1))
        return True
