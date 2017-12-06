'''
2. Add Two Numbers
'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

'''
0 0 1
2
-> 2 0 1

1
2
-> 3

5 5
6 6
-> 1 2 1

1
None
-> 1
'''

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        def appendDigit(n, x):
            newNode = ListNode(x)
            n.next = newNode
            return newNode

        dummyHead = ListNode(0)
        curr = dummyHead
        carryOver = 0
        while l1 or l2:
            tempSum = carryOver
            tempSum += l1.val if l1 else 0
            tempSum += l2.val if l2 else 0
            carryOver, digit = divmod(tempSum, 10)
            curr = appendDigit(curr, digit)
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
        if carryOver:
            curr = appendDigit(curr, carryOver)

        return dummyHead.next

