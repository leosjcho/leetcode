'''
104. Maximum Depth of Binary Tree
'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        def maxDepthHelper(root, depth):
            if not root:
                return depth
            return max(maxDepthHelper(root.left, depth+1),
                       maxDepthHelper(root.right, depth+1))
        return maxDepthHelper(root, 0)

