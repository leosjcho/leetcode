'''
1. Two Sum
'''

class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        complements = {}
        for i, x in enumerate(nums):
            complement = target - x
            if complement in complements:
                return [complements[complement], i]
            else:
                complements[x] = i


