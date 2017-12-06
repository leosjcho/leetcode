'''
151. Reverse Words in a String
'''

class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        s = " ".join(s.split())
        def reverse(s, l, r):
            while l < r:
                s[l], s[r] = s[r], s[l]
                l += 1
                r -= 1

        s = list(s)
        reverse(s, 0, len(s)-1)
        word_start_index = 0
        for i in range(len(s)+1):
            if i == len(s) or s[i] == ' ':
                reverse(s, word_start_index, i-1)
                word_start_index = i + 1

        return "".join(s)

