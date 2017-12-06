/*
148. Sort List
*/

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func sortList(head *ListNode) *ListNode {
	// get "head" of right half
	if head == nil || head.Next == nil {
		return head
	}
	rightHalfHead := getRightHalfHead(head)
	leftHead := sortList(head)
	rightHead := sortList(rightHalfHead)
	return merge(leftHead, rightHead)
}

func getRightHalfHead(head *ListNode) *ListNode {
	var stepperParent *ListNode
	stepper, jumper := head, head
	for jumper != nil && jumper.Next != nil {
		stepperParent = stepper
		stepper = stepper.Next
		jumper = jumper.Next.Next
	}
	stepperParent.Next = nil
	return stepper
}

func merge(left *ListNode, right *ListNode) *ListNode {
	l := &ListNode{}
	p := l
	for left != nil && right != nil {
		if left.Val < right.Val {
			p.Next = left
			left = left.Next
		} else {
			p.Next = right
			right = right.Next
		}
		p = p.Next
	}
	if left != nil {
		p.Next = left
	}
	if right != nil {
		p.Next = right
	}
	return l.Next
}

