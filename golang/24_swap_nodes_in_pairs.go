/*
24. Swap Nodes in Pairs
*/

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func swapPairs(head *ListNode) *ListNode {
	if head == nil {
		return head
	}
	if head.Next == nil {
		return head
	}
	child := head.Next
	head.Next = swapPairs(head.Next.Next)
	child.Next = head
	return child
}

