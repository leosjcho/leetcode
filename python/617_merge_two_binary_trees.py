'''
617. Merge Two Binary Trees
'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def mergeTrees(self, t1, t2):
        """
        :type t1: TreeNode
        :type t2: TreeNode
        :rtype: TreeNode
        """
        if not t1 and not t2:
            return None
        if t2 and not t1:
            t1, t2 = t2, t1
        elif t1 and t2:
            t1.val = t1.val + t2.val

        t1.left = self.mergeTrees(t1.left, t2.left if t2 else None)
        t1.right = self.mergeTrees(t1.right, t2.right if t2 else None)
        return t1
