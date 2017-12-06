/*
2. Add Two Numbers
*/

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */

// while l1 && l2 exist
// sum = l1 and l2 and carryover
// carryover = sum // 10 (integer division)
// sum %= 10
// add new node to sumlist with sum
// while l1 exists
// sum = l1 + carryover
// perform same operation as above
// while l2 exists
// perform same operations as above
// if carryover > 0
// add new node to sumlist with carryover

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	carryover := 0
	sumHead := &ListNode{}
	sumTail := sumHead
	for l1 != nil || l2 != nil || carryover > 0 {
		l1v, l2v := 0, 0
		if l1 != nil {
			l1v = l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			l2v = l2.Val
			l2 = l2.Next
		}
		sum := l1v + l2v + carryover
		n := &ListNode{sum % 10, nil}
		carryover = sum / 10
		sumTail.Next = n
		sumTail = n
	}
	return sumHead.Next
}

/*
l1 = [2]
l2 = [9, 5]
carryover = 0
sumhead = empty node
sumtail = sumhead
l1 and l2 not nil
sum = 2 + 9 + 0 = 11
c = 11 / 10 = 1
sum = 1
sumtail.next = {1, nil}
carryover = 1, sumTail = {1, nil}
l1 = nil, l2 = 5
sum = 0 + 5 + 1 = 6
c = 6 / 10 = 0
node = {6, nil}
output:
1 -> 6
*/

