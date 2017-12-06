/*
53. Maximum Subarray
*/

/*
Find the contiguous subarray within an array (containing at least one number)
which has the largest sum.

For example, given the array [-2,1,-3,4,-1,2,1,-5,4],
the contiguous subarray [4,-1,2,1] has the largest sum = 6.
*/

/*
// -2 1 -3 4
// gm = -inf lm = -inf
-2:
lm = -2
gm = -2
1:
lm = max(-2 + 1, 1) = 1
gm = max(-2, 1) = 1
-3:
lm = max(1 - 3, -3) = -2
gm = max(1, -2) = 1
4:
lm = max(-2+4, 4) = 4
gm = max(1, 4) = 4
return 4
*/

func maxSubArray(nums []int) int {
	globalMax, localMax := math.MinInt32, math.MinInt32
	for _, x := range nums {
		localMax = max(localMax+x, x)
		globalMax = max(globalMax, localMax)
	}
	return globalMax
}

