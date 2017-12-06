/*
206. Reverse Linked List
*/

type ListNode struct {
	Val  int
	Next *ListNode
}

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func reverseList(head *ListNode) *ListNode {
	if head == nil {
		return head
	}
	return reverseListHelper(head, nil)
}

func reverseListHelper(node *ListNode, p *ListNode) *ListNode {
	next := node.Next
	node.Next = p
	if next == nil {
		return node
	}
	return reverseListHelper(next, node)
}

