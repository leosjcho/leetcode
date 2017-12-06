'''
100. Same Tree
'''

def is_equal(t1, t2):
    stack1, stack2 = [t1], [t2]
    while stack1 and stack2:
        n1, n2 = stack1.pop(), stack2.pop()
        if (n1 and not n2) or (not n1 and n2):
            return False
        if n1 and n2 and n1.val != n2.val:
            return False
        if n1 and n2:
            stack1.extend([n1.left, n1.right])
            stack2.extend([n2.left, n2.right])
    return stack1 == stack2

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if p and q:
            return p.val == q.val and self.isSameTree(p.left, q.left) and \
                self.isSameTree(p.right, q.right))
        return p is q

