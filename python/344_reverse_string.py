'''
344. Reverse String
'''

class Solution(object):
    def reverseString(self, s):
        """
        :type s: str
        :rtype: str
        """
        A = list(s)
        l, r = 0, len(s)-1
        while l < r:
            A[l], A[r] = A[r], A[l]
            l += 1
            r -= 1
        return "".join(A)

