'''
152. Maximum Product Subarray
'''

class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        curMax = curMin = maxSeen = nums[0]
        for x in nums[1:]:
            temp = curMax
            curMax = max(temp * x, curMin * x, x)
            curMin = min(temp * x, curMin * x, x)
            maxSeen = max(maxSeen, curMax)
        return maxSeen

