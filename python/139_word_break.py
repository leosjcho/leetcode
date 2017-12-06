'''
139. Word Break
'''

class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        cache = {}

        def isWord(s):
            return s in wordDict

        def isPartionable(i):
            if i == len(s):
                return True
            if i in cache:
                return cache[i]
            for j in range(i, len(s)):
                candidate_word = s[i:j+1]
                if isWord(candidate_word) and isPartionable(j+1):
                    cache[i] = True
                    return cache[i]
            cache[i] = False
            return cache[i]

        return isPartionable(0)

