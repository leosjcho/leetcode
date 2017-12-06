'''
88. Merge Sorted Array
'''

class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        i = m + n - 1
        m -= 1
        n -= 1
        while i >= 0 and m >= 0 and n >= 0:
            if nums1[m] < nums2[n]:
                maxVal = nums2[n]
                n -= 1
            else:
                maxVal = nums1[m]
                m -= 1
            nums1[i] = maxVal
            i -= 1
        while m >= 0:
            nums1[i] = nums1[m]
            i -= 1
            m -= 1
        while n >= 0:
            nums1[i] = nums2[n]
            i -= 1
            n -= 1

