/*
167. Two Sum II - Input array is sorted
*/

func twoSum(numbers []int, target int) []int {
	n := len(numbers)
	i, j := 0, n-1
	for {
		sum := numbers[i] + numbers[j]
		if sum == target {
			return []int{i + 1, j + 1}
		} else if sum > target {
			j--
		} else {
			i++
		}
	}
}

