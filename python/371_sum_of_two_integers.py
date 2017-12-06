'''
371. Sum of Two Integers
'''

class Solution(object):
    def getSum(self, a, b):
        """
        :type a: int
        :type b: int
        :rtype: int
        """
        # 32 bits integer max
        MAX = 0x7FFFFFFF
        # 32 bits interger min
        MIN = 0x80000000
        # mask to get last 32 bits
        mask = 0xFFFFFFFF
        carryover = b
        while carryover != 0:
            digits = (a ^ b)
            carryover = (a & b) << 1
            a = digits & mask
            b = carryover & mask
        return a if a <= MAX else ~(a ^ mask)

