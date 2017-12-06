'''
384. Shuffle an Array
'''

import random

class Solution(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.nums = nums

    def reset(self):
        """
        Resets the array to its original configuration and return it.
        :rtype: List[int]
        """
        return self.nums

    def shuffle(self):
        """
        Returns a random shuffling of the array.
        :rtype: List[int]
        """
        shuffled = self.nums[:]
        result_index = 0
        while result_index < len(self.nums) - 1:
            rand_index = random.randrange(result_index, len(self.nums))
            shuffled[result_index], shuffled[rand_index] = \
                shuffled[rand_index], shuffled[result_index]
            result_index += 1
        return shuffled

