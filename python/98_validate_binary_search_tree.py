'''
98. Validate Binary Search Tree
'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.isValidBSTHelper(root, float('-inf'), float('inf'))

    def isValidBSTHelper(self, root, minval, maxval):
        if not root:
            return True
        if root.val <= minval or root.val >= maxval:
            return False
        return (self.isValidBSTHelper(root.left, minval, root.val)
                and self.isValidBSTHelper(root.right, root.val, maxval))

