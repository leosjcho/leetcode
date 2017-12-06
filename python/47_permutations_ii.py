'''
47. Permutations II
'''

class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if len(nums) == 1:
            return [nums]
        output = []
        seen = {}
        for i in range(len(nums)):
            if nums[i] not in seen:
                seen[nums[i]] = True
                nums[0], nums[i] = nums[i], nums[0]
                for p in self.permuteUnique(nums[1:]):
                    output.append([nums[0]] + p)
                nums[0], nums[i] = nums[i], nums[0]
        return output

