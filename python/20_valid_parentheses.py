'''
20. Valid Parentheses
'''

class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        brackets = {")":"(","}":"{","]":"["}
        for char in s:
            if char in brackets.values():
                stack.append(char)
            elif char in brackets.keys():''
                if len(stack) > 0 and stack[-1] == brackets[char]:
                    stack.pop()
                else:
                    return False
        return len(stack) == 0

