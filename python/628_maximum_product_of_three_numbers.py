'''
628. Maximum Product of Three Numbers
'''

import heapq
from functools import reduce
from operator import mul

class Solution(object):
    def maximumProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        [1, 2, 3] -> trivial, return all 3 numbers multiplied
        [7, 6, 2, 4]
        [0, 6, 2, 4]
        things to think about ... negative numbers and zeroes
        brute force is n * n * n O(n^3)
        max product will be
        three largest positive numbers
        two largest negative and largest positive
            largest negative and two positive leads to negative #
            three largest negative leads to negative
        maintain min and max heaps
        pop off smallest two and largest 3, find max between the two
        '''
        max_3, min_2 = nums[:3], list(map(lambda x: -x, nums[:2]))
        heapq.heapify(max_3)
        heapq.heapify(min_2)
        heapq.heappushpop(min_2, -nums[2])
        for i, x in enumerate(nums[3:]):
            heapq.heappushpop(max_3, x)
            heapq.heappushpop(min_2, -x)
        # print(max_3, min_2)
        pos_3 = functools.reduce(lambda t, x: t * x, max_3)
        neg_2 = -min_2[0] * -min_2[1]
        neg_2 *= max(max_3)
        # print(pos_3, neg_2)
        return max(pos_3, neg_2)

        '''
        concise solution
        '''

        max_3, min_2 = heapq.nlargest(3, nums), heapq.nsmallest(2, nums)
        return max(reduce(mul, max_3), reduce(mul, min_2 + [max_3[0]]))

