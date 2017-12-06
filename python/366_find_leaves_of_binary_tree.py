'''
366. Find Leaves of Binary Tree
'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def findLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return None
        output = []
        q = collections.deque()
        q.append((root, 0))
        while q:
            n, d = q.popleft()
            if not n:
                continue
            if d >= len(output):
                output.append([])
            output[d].append(n.val)
            q.append((n.left, d+1))
            q.append((n.right, d+1))
        return output.reverse()

