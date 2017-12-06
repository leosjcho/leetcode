'''
11. Container With Most Water
'''

class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        i, j = 0, len(height)-1
        maxArea = 0
        while i < j:
            area = min(height[i], height[j]) * (j - i)
            maxArea = max(maxArea, area)
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
        return maxArea

