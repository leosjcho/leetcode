'''
160. Intersection of Two Linked Lists
'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        initialA, initialB = headA, headB
        aLooped = bLooped = False
        while headA and headB:

            if headA.val == headB.val:
                return headA

            headA = headA.next
            headB = headB.next

            if not headA and not aLooped:
                headA = initialB
                aLooped = True
            if not headB and not bLooped:
                headB = initialA
                bLooped = True

        return None
    
