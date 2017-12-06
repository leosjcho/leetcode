package golang

/*
238. Product of Array Except Self
*/

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
