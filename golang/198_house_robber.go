/*
198. House Robber
*/

func rob(nums []int) int {
	prev, curr := 0, 0
	for _, p := range nums {
		temp := curr
		curr = max(prev+p, curr)
		prev = temp
	}
	return curr
}

