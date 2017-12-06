/*
283. Move Zeroes
*/

func moveZeroes(nums []int) {
	// track zeros seen
	zeroesSeen := 0
	// iterate over array
	for i, x := range nums {
		if x == 0 {
			zeroesSeen++
		} else {
			newPos := i - zeroesSeen
			nums[newPos] = x
			if zeroesSeen > 0 {
				nums[i] = 0
			}
		}
	}
}

