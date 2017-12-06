/*
238. Product of Array Except Self
*/

// 1 2 3 4
// 2 * 3 * 4, 1 * 3 * 4, 1 * 2 * 4, 1 * 2 * 3

/*
func productExceptSelf(nums []int) []int {
	res := []int{1}
	for i := 1; i < len(nums); i++ {
		res = append(res, res[i-1]*nums[i-1])
	}
	right := 1
	for j := len(nums) - 1; j >= 0; j-- {
		res[j] *= right
		right *= nums[j]
	}
	return res
}
*/

// explcit
func productExceptSelf(nums []int) []int {
	results := make([]int, len(nums))
	runningTotal := 1
	for i := range nums {
		results[i] = runningTotal
		runningTotal *= nums[i]
	}
	runningTotal = 1
	for i := len(nums) - 1; i >= 0; i-- {
		results[i] *= runningTotal
		runningTotal *= nums[i]
	}
	return results
}

