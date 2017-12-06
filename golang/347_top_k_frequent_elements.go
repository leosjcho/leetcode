/*
347. Top K Frequent Elements
*/

/*
Given a non-empty array of integers, return the k most frequent elements.

For example,
Given [1,1,1,2,2,3] and k = 2, return [1,2].
*/

// count occurances of each number in hash table
// then build max heap with (num, occurances) tuples

func topKFrequent(nums []int, k int) []int {
	occurances := countOccurances(nums) // returns map[int]int
	// build max heap with occurances
	pq := NewHeap(occurances)
	top := []int{}
	for i := 0; i < k; i++ {
		top = append(top, heap.Pop(pq).(*KFreqItem).num)
	}
	return top
}

func countOccurances(nums []int) map[int]int {
	counts := map[int]int{}
	for _, v := range nums {
		counts[v]++
	}
	return counts
}

type KFreqItem struct {
	num   int
	count int
}

type KFreqPQ []*KFreqItem

func (pq KFreqPQ) Len() int {
	return len(pq)
}

func (pq KFreqPQ) Less(i, j int) bool {
	return pq[i].count > pq[j].count
}

func (pq KFreqPQ) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}

func (pq *KFreqPQ) Push(x interface{}) {
	*pq = append(*pq, x.(*KFreqItem))
}

func (pq *KFreqPQ) Pop() interface{} {
	n := len(*pq)
	old := *pq
	item := old[n-1]
	*pq = old[:n-1]
	return item
}

func NewHeap(occurances map[int]int) *KFreqPQ {
	pq := make(KFreqPQ, 0)
	for k, v := range occurances {
		pq = append(pq, &KFreqItem{k, v})
	}
	heap.Init(&pq)
	return &pq
}

