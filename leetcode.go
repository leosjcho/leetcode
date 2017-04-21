package main

import "strings"

/*
461. Hamming Distance
*/

func hammingDistance(x int, y int) int {
	count := 0
	var i uint
	xor := uint(x) ^ uint(y)
	for i = 0; i < 32; i++ {
		if ((1 << i) & xor) != 0 {
			count++
		}
	}
	return count
}

/*
557. Reverse Words in a String III
*/

func reverseWords(s string) string {
	rw := []string{}
	ws := strings.Split(s, " ")
	for _, w := range ws {
		rw = append(rw, reverseWord(w))
	}
	return strings.Join(rw, " ")
}

func reverseWord(s string) string {
	b := []byte(s)
	for i := 0; i < len(b)/2; i++ {
		b[i], b[len(b)-1-i] = b[len(b)-1-i], b[i]
	}
	return string(b)
}

/*
26. Remove Duplicates from Sorted Array
*/

func removeDuplicates(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	i := 0
	for j := 0; j < len(nums); j++ {
		if nums[i] != nums[j] {
			i++
			nums[i] = nums[j]
		}
	}
	return i + 1
}

/*
167. Two Sum II - Input array is sorted
*/

/*
two pointer
if i is less than j, stop
if val[i] == val[j] stop
1 2 4 4 7 target: 6
1 + 7 = 8 - too large, decrement j
1 + 4 = 5 - too small, increment i
2 + 4 = 6 - hooray!
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

/*
189. Rotate Array
*/

/*
1 2 3
rotate 1
3 1 2
rotate 2
2 3 1
rotate 3
1 2 3
k = k % n

*/

func rotate(nums []int, k int) {
	n := len(nums)
	k, count := k%n, 0
	for start := 0; count < n; start++ {
		current, temp := start, nums[start]
		for {
			current = (current + k) % n
			nums[current], temp = temp, nums[current]
			count++
			if start == current {
				break
			}
		}
	}
}
