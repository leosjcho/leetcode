'''
280. Wiggle Sort
'''

class Solution(object):
    def wiggleSort(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """

        def swap(A, i, j):
            A[i], A[j] = A[j], A[i]

        shouldBeLessThan = True
        for i in range(len(nums)-1):
            if shouldBeLessThan:
                if nums[i] > nums[i+1]:
                    swap(nums, i, i+1)
                shouldBeLessThan = False
            else:
                if nums[i] < nums[i+1]:
                    swap(nums, i, i+1)
                shouldBeLessThan = True

