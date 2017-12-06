'''
346. Moving Average from Data Stream
'''

class MovingAverage(object):

    def __init__(self, size):
        """
        Initialize your data structure here.
        :type size: int
        """
        self._circularArray = [0 for i in range(size)]
        self._counter = 0
        self._sum = 0
        self._currIndex = 0

    def next(self, val):
        """
        :type val: int
        :rtype: float
        """
        self._counter += 1
        self._sum -= self._circularArray[self._currIndex]
        self._sum += val
        self._circularArray[self._currIndex] = val
        self._currIndex = (self._currIndex + 1) % len(self._circularArray)
        return float(self._sum) / min(len(self._circularArray), self._counter)

# Your MovingAverage object will be instantiated and called as such:
# obj = MovingAverage(size)
# param_1 = obj.next(val)

