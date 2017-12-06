'''
101. Symmetric Tree
'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def traversal(t1, t2):
            if bool(t1) != bool(t2) or ((t1 and t2) and (t1.val != t2.val)):
                return False
            if not t1 and not t2:
                return True
            if not traversal(t1.left, t2.right) or \
                not traversal(t1.right, t2.left):
                    return False
            return True
        if root:
            return traversal(root.left, root.right)
        return True

        '''
        EPI solution
        '''

        def check_symmetric(t1, t2):
            if not t1 and not t2:
                return True
            elif t1 and t2:
                return t1.val == t2.val and \
                    check_symmetric(t1.left, t2.right) and \
                    check_symmetric(t1.right, t2.left)
            return False
        return not root or check_symmetric(root.left, root.right)

