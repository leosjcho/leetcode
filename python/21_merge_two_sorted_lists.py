'''
21. Merge Two Sorted Lists
'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        '''
        # recursive
        if not l1 or not l2:
            return l1 or l2
        m = ListNode(0) # dummy head
        if l1.val < l2.val:
            m.next = l1
            m.next.next = self.mergeTwoLists(l1.next, l2)
        else:
            m.next = l2
            m.next.next = self.mergeTwoLists(l1, l2.next)
        return m.next
        '''
        # iterative
        m = ListNode(0)
        tail = m
        while l1 and l2:
            if l1.val < l2.val:
                tail.next, l1 = l1, l1.next
            else:
                tail.next, l2 = l2, l2.next
            tail = tail.next
        tail.next = l1 or l2
        return m.next


