'''
236. Lowest Common Ancestor of a Binary Tree
'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

import collections

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        Status = collections.namedtuple("Status", ('num_nodes', 'ancestor'))

        def lca_helper(tree, n1, n2):
            if not tree:
                return Status(0, None)
            left = lca_helper(tree.left, n1, n2)
            if left.num_nodes == 2:
                return left
            right = lca_helper(tree.right, n1, n2)
            if right.num_nodes == 2:
                return right
            num_nodes = left.num_nodes + right.num_nodes + \
                int(tree is n1) + int(tree is n2)
            return Status(num_nodes, tree if num_nodes == 2 else None)

        return lca_helper(root, p, q).ancestor

