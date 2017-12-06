'''
300. Longest Increasing Subsequence
'''

class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        lengths = [0 for _ in nums]
        for i, x in enumerate(nums):
            lengths[i] = max(
                [lengths[j] if nums[j] < x else 0 for j in range(i)] + [0]
            ) + 1
        return max(lengths)

