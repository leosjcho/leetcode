/*
169. Majority Element
*/

// O(n) time complexity, O(n) space complexity
func majorityElement(nums []int) int {
	counts := map[int]int{}
	maxElement, maxCount := 0, 0
	for _, num := range nums {
		counts[num]++
		if counts[num] > maxCount {
			maxElement = num
			maxCount = counts[num]
		}
	}
	return maxElement
}

