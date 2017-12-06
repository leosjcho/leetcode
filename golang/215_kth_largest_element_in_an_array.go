/*
215. Kth Largest Element in an Array
*/

// modification of quicksort in descending order (largest to smallest)
// given array [5, 2, 3, 1, 4] and k = 2, we want 4
// pick random element, move larger elements to left,
// smaller elements to right.
// our random element will end up in position i
// if i == k-1, then we're done. that's the value! how lucky.
// if i < k-1, then recursively perform this modified quicksort on
// the right side of array
// if i > k-1, perform on left side of array
func findKthLargest(nums []int, k int) int {
	lo, hi := 0, len(nums)-1
	for lo <= hi {
		i := partition(nums, lo, hi)
		if i == k-1 {
			return nums[i]
		} else if i < k-1 {
			lo = i + 1
		} else { // i is > k-1
			hi = i - 1
		}
	}
	return -1 // error code
}

func partition(nums []int, lo, hi int) int {
	i := lo - 1
	pivot := nums[hi]
	for j := lo; j < hi; j++ {
		if nums[j] >= pivot {
			i++
			nums[i], nums[j] = nums[j], nums[i]
		}
	}
	nums[i+1], nums[hi] = nums[hi], nums[i+1]
	return i + 1
}

