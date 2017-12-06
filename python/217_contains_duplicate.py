'''
217. Contains Duplicate
'''

from collections import defaultdict
class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        seen = defaultdict(int)
        for x in nums:
            seen[x] += 1
            if seen[x] > 1:
                return True
        return False

