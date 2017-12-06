'''
22. Generate Parentheses
'''

class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        validArrangements = []
        def backtrack(string, opens, closes):
            if len(string) == n*2:
                validArrangements.append("".join(string))
                return
            if opens < n:
                backtrack(string+["("], opens+1, closes)
            if closes < opens:
                backtrack(string+[")"], opens, closes+1)
        backtrack([], 0, 0)
        return validArrangements

