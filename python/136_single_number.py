'''
136. Single Number
'''

import functools

class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return functools.reduce(lambda t, x: t ^ x, nums)

