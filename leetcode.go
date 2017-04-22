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

/*
125. Valid Palindrome
*/

func isPalindrome(s string) bool {
	ss := []rune{}
	s = strings.ToLower(s)
	for _, c := range s {
		if ((c >= 'a') && (c <= 'z')) || ((c >= '0') && (c <= '9')) {
			ss = append(ss, c)
		}
	}
	i, j := 0, len(ss)-1
	for {
		if j <= i {
			return true
		}
		if ss[i] != ss[j] {
			return false
		}
		j--
		i++
	}
}

/*
11. Container With Most Water
*/

func maxArea(height []int) int {
	i, j := 0, len(height)-1
	maxArea := 0
	for {
		if j <= i {
			return maxArea
		}
		min := min(height[i], height[j])
		area := min * (j - i)
		maxArea = max(maxArea, area)
		if height[i] < height[j] {
			i++
		} else {
			j--
		}
	}
}

func min(x, y int) int {
	if y < x {
		return y
	}
	return x
}

func max(x, y int) int {
	if y > x {
		return y
	}
	return x
}

/*
238. Product of Array Except Self
*/

// 1 2 3 4
// 2 * 3 * 4, 1 * 3 * 4, 1 * 2 * 4, 1 * 2 * 3

func productExceptSelf(nums []int) []int {
	res := []int{1}
	for i := 1; i < len(nums); i++ {
		res = append(res, res[i-1]*nums[i-1])
	}
	right := 1
	for j := len(nums) - 1; j >= 0; j-- {
		res[j] *= right
		right *= nums[j]
	}
	return res
}

/*
242. Valid Anagram
*/

func isAnagram(s string, t string) bool {
	chars := map[rune]int{}
	for _, c := range s {
		chars[c]++
	}
	for _, c := range t {
		chars[c]--
	}
	for _, count := range chars {
		if count != 0 {
			return false
		}
	}
	return true
}
