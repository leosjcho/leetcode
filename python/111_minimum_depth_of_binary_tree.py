'''
111. Minimum Depth of Binary Tree
'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

import collections

class Solution(object):
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        q = collections.deque()
        q.append((root, 1))
        while q:
            n, d = q.popleft()
            if not n.left and not n.right:
                return d
            if n.left:
                q.append((n.left, d+1))
            if n.right:
                q.append((n.right, d+1))

