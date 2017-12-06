'''
448. Find All Numbers Disappeared in an Array
'''

class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """

        '''
        num_map = {}
        missing_nums = []
        for x in nums:
            num_map[x] = 1
        for i in range(1, len(nums) + 1):
            if i not in num_map:
                missing_nums.append(i)
        return missing_nums
        '''

        # For each number i in nums,
        # we mark the number that i points as negative.
        # Then we filter the list, get all the indexes
        # who points to a positive number
        for i in range(len(nums)):
            index = abs(nums[i]) - 1
            nums[index] = - abs(nums[index])

        return [i + 1 for i in range(len(nums)) if nums[i] > 0]


