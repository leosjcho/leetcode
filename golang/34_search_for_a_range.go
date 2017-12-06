/*
34. Search for a Range
*/

func searchRange(nums []int, target int) []int {
	left := search(nums, target, func(target int, nums []int, mid int) bool {
		return target == nums[mid] && (mid == 0 || target != nums[mid-1])
	}, true)
	right := search(nums, target, func(target int, nums []int, mid int) bool {
		return target == nums[mid] && (mid == len(nums)-1 || target != nums[mid+1])
	}, false)
	return []int{left, right}
}

func search(nums []int, target int, success func(target int, nums []int,
	mid int) bool,
	checkLeft bool) int {
	lo, hi := 0, len(nums)-1
	for lo <= hi {
		mid := ((hi - lo) / 2) + lo
		if success(target, nums, mid) {
			return mid
		}
		if target < nums[mid] || ((target == nums[mid]) && checkLeft) {
			hi = mid - 1
		} else {
			lo = mid + 1
		}
	}
	return -1
}

