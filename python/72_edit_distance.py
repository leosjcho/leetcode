'''
72. Edit Distance
'''

class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        def score(i, j):
            return 0 if word1[i-1] == word2[j-1] else 1
        n, m = len(word1)+1, len(word2)+1
        dp = [ [i] + [j if i == 0 else 0 for j in range(1, m)] \
            for i in range(n)]
        for i in range(1, n):
            for j in range(1, m):
                dp[i][j] = min(
                    dp[i-1][j-1] + score(i,j),
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1
                )
        return dp[n-1][m-1]

