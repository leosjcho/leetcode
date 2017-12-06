/*
35. Search Insert Position
*/

func searchInsert(nums []int, target int) int {
	lo, hi := 0, len(nums)-1
	for lo <= hi {
		mid := ((hi - lo) / 2) + lo
		if nums[mid] == target {
			return mid
		}
		if target < nums[mid] {
			hi = mid - 1
		} else {
			lo = mid + 1
		}
	}
	return lo
}

