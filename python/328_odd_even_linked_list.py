'''
328. Odd Even Linked List
'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        oddDummy = oddTail = ListNode(0)
        evenDummy = evenTail = ListNode(0)
        while head:
            oddTail.next = head
            oddTail = oddTail.next
            evenTail.next = head.next
            evenTail = evenTail.next
            head = head.next.next if head.next else None
        oddTail.next = evenDummy.next
        return oddDummy.next

