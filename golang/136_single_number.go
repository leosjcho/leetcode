/*
136. Single Number
*/

func singleNumber(nums []int) int {
	v := nums[0]
	for i := 1; i < len(nums); i++ {
		v = v ^ nums[i]
	}
	return v
}

