'''
141. Linked List Cycle
'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if not head:
            return False
        n1, n2 = head, head.next
        while n1 and n2:
            if n1.val == n2.val:
                return True
            n1 = n1.next
            if n2.next:
                n2 = n2.next.next
            else:
                return False
        return False

