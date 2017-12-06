'''
155. Min Stack
'''

class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.items = []
        self.minstack = []

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        self.items.append(x)
        if not self.minstack.peek() or x <= self.minstack.peek():
            self.minstack.append(x)

    def pop(self):
        """
        :rtype: void
        """
        if not self.isEmpty():
            x = self.items.pop()
            if x == self.minstack[-1]:
                self.minstack.pop()
            return x

    def top(self):
        """
        :rtype: int
        """
        if not self.isEmpty():
            return self.items[-1]


    def getMin(self):
        """
        :rtype: int
        """
        if not self.isEmpty():
            return self.minstack[-1]

    def isEmpty(self):
        return not self.items

# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()

