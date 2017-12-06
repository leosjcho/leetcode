'''
110. Balanced Binary Tree
'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        height = self.isBalancedHelper(root, 0)
        return height >= 0

    def isBalancedHelper(self, root, height):
        '''
        :type root: TreeNode, height: int
        :rtype height: int
        '''
        if not root:
            return height
        lh = self.isBalancedHelper(root.left, height+1)
        rh = self.isBalancedHelper(root.right, height+1)
        if lh == -1 or rh == -1 or abs(lh-rh) > 1:
            return -1
        return max(lh, rh)

