'''
461. Hamming Distance
'''

class Solution(object):
    def hammingDistance(self, x, y):
        """
        :type x: int
        :type y: int
        :rtype: int
        """
        xor = x ^ y
        diffBits = 0
        while xor:
            xor = xor & (xor - 1)
            diffBits += 1
        return diffBits

