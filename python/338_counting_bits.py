'''
338. Counting Bits
'''

class Solution(object):
    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """
        cache = {}
        def countBitsHelper(num):
            if num == 0:
                return 0
            if num in cache:
                return cache[num]
            cache[num] = countBitsHelper((num & (num - 1))) + 1
            return cache[num]

        return [countBitsHelper(i) for i in range(num+1)]


