'''
53. Maximum Subarray
'''

class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return None
        maxSum, curSum = float('-inf'), 0
        for x in nums:
            curSum += x
            maxSum = max(maxSum, curSum)
            if curSum < 0:
                curSum = 0
        return maxSum

