'''
102. Binary Tree Level Order Traversal
'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

import collections

class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        output = []
        if not root:
            return output
        Item = collections.namedtuple('Item', ('node', 'depth'))
        q = collections.deque()
        q.append(Item(root, 0))
        while q:
            x = q.popleft()
            node = x.node
            if node:
                depth = x.depth
                if depth not in range(len(output)):
                    output.append([])
                output[depth].append(node.val)
                q.append(Item(node.left, depth + 1))
                q.append(Item(node.right, depth + 1))
        return output

