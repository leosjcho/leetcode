/*
23. Merge k Sorted Lists
*/

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */

type PQItem struct {
	node  *ListNode
	index int
}

type PriorityQueue []*PQItem

func (pq PriorityQueue) Len() int {
	return len(pq)
}

func (pq PriorityQueue) Less(i, j int) bool {
	return pq[i].node.Val < pq[j].node.Val
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}

func (pq *PriorityQueue) Push(x interface{}) {
	*pq = append(*pq, x.(*PQItem))
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	item := old[len(*pq)-1]
	*pq = old[0 : len(*pq)-1]
	return item
}

func heapify(lists []*ListNode) *PriorityQueue {
	pq := make(PriorityQueue, 0)
	for i := range lists {
		if lists[i] != nil {
			pq = append(pq, &PQItem{lists[i], i})
		}
	}
	heap.Init(&pq)
	return &pq
}

func mergeKLists(lists []*ListNode) *ListNode {
	// build heap with list heads
	pq := heapify(lists) // O(n log n)
	curr := &ListNode{}
	headPointer := &ListNode{Next: curr} // pointer to head
	// while heap is not empty
	for pq.Len() > 0 {
		// get node and it's index in the list of lists
		item := heap.Pop(pq).(*PQItem) // O(log n)
		node, i := item.node, item.index
		lists[i] = node.Next
		if lists[i] != nil {
			newItem := &PQItem{lists[i], i}
			heap.Push(pq, newItem)
		}
		curr.Next = node
		curr = node
	}
	return headPointer.Next.Next
}

// overall runtime O(n log n)
// space complexity O(n)

