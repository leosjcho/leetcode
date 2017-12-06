'''
14. Longest Common Prefix
'''

class TrieNode:
    def __init__(self, x):
        self.val = x
        self.children = {}
        self.count = 0

    def childNode(self):
        if len(self.children) == 1
            return self.children[list(self.children.keys())[0]]
        else:
            return None

class Trie:
    def __init__(self):
        self.root = TrieNode(0)

    def insert(self, s):
        cur = self.root
        for i, c in enumerate(s):
            if c not in cur.children:
                cur.children[c] = TrieNode(c)
            cur = cur.children[c]
            cur.count += 1

    def lcp(self, root, num_strs):
        if not root or root.count != num_strs:
            return ""
        output = [root.val]
        if len(root.children) == 1:
            output.extend(self.lcp(root.childNode(), num_strs))
        return output

class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if len(strs) == 1:
            return strs[0]
        t = Trie()
        for s in strs:
            t.insert(s)
        output = t.lcp(t.root.childNode(), len(strs))
        return "".join(output) if output else ""

'''
simplified vertical scanning solution
'''

class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if not strs:
            return ""
        if len(strs) == 1:
            return strs[0]
        base = strs[0]
        for i, c in enumerate(base):
                for s in strs:
                    if i == len(s) or s[i] != c:
                        return base[:i]
        return base

