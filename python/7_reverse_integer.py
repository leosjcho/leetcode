'''
7. Reverse Integer
'''

class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        max32 = 0x7fffffff
        rev = 0
        isNeg = x < 0
        x = abs(x)
        while x != 0:
            rev, x = rev * 10 + x % 10, x // 10
        if rev > max32:
            return 0
        return rev * -1 if isNeg else rev

