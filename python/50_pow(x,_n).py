'''
50. Pow(x, n)
'''

class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        result, power = 1.0, n
        if power < 0:
            power, x = -power, 1.0 / x
        while power:
            if power & 1:
                result *= x
            x *= x
            power >>= 1
        return result

