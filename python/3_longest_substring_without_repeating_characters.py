'''
3. Longest Substring Without Repeating Characters
'''

class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        start = curmax = 0
        seen = {}
        for i, c in enumerate(s):
            if c in seen and start <= seen[c]:
                start = seen[c] + 1
            else:
                curmax = max(curmax, i - start + 1)
            seen[c] = i
        return curmax


