/*
1. Two Sum
*/

// name modified to prevent atom errors
func twoSumREMOVEME(nums []int, target int) []int {
	// complement map
	// keep track of complements seen
	// linear time and space complexity
	complements := map[int]int{}
	for i, x := range nums {
		complement := target - x
		j, ok := complements[complement]
		if ok {
			return []int{j, i}
		} else {
			complements[x] = i
		}
	}
	return []int{}
}

// try with 1 2 3 4
// target = 5
// 0, 1. complement = 5 - 1 = 4
// complements[4] == nil
// complements[1] = 0
// i = 1, x = 2. complement = 5 - 2 = 3
// complements[3] == nil
// complements[2] = 1
// i = 2, x = 3. complements = 5-3=2
// complements [2] = 1
// returns [1, 2]

