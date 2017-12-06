'''
27. Remove Element
'''

class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        swap_index = len(nums)-1
        swap_count = 0
        i = 0
        while i <= swap_index:
            if nums[i] == val:
                nums[i], nums[swap_index] = nums[swap_index], nums[i]
                swap_index -= 1
                swap_count += 1
            else:
                i += 1
        return len(nums) - swap_count

