'''
198. House Robber
'''

class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        if len(nums) <= 1:
            return nums[0]
        if len(nums) <= 2:
            return max(nums[0], nums[1])
        # have at least 3 elements in the array at this point
        dp = [x for x in nums]
        for i, x in enumerate(nums[2:], 2):
            # skip current and take previous
            # if taking current, you're
            dp[i] = max(dp[i-1],
                        dp[i-2]+x,
                        dp[i-3]+x if i >= 3 else float('-inf'))
        return dp[len(nums)-1]

