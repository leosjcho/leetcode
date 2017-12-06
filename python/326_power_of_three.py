'''
326. Power of Three
'''

from math import *

class Solution(object):
    def isPowerOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        """
        return (log10(n)/log10(3)).is_integer() if n > 0 else False

