package main

import (
	"math"
	"strconv"
	"strings"
	"unicode"
)

/*
Data Structures
*/

type Queue struct {
	items []int
}

func NewQueue() *Queue {
	return &Queue{items: []int{}}
}

func (q *Queue) Push(x int) {
	q.items = append(q.items, x)
}

func (q *Queue) Peek() int {
	if q.Size() == 0 {
		return -1
	}
	return q.items[0]
}

func (q *Queue) Pop() int {
	if q.Size() == 0 {
		return -1
	}
	x := q.items[0]
	q.items = q.items[1:]
	return x
}

func (q *Queue) Size() int {
	return len(q.items)
}

func (q *Queue) IsEmpty() bool {
	return len(q.items) == 0
}

type AStack struct {
	items []string
}

func NewAStack() *AStack {
	return &AStack{[]string{}}
}

func (a *AStack) Push(s string) {
	a.items = append(a.items, s)
}

func (a *AStack) Pop() string {
	if len(a.items) == 0 {
		return ""
	}
	val := a.items[len(a.items)-1]
	a.items = a.items[:len(a.items)-1]
	return val
}

func (a *AStack) Peek() string {
	if len(a.items) == 0 {
		return ""
	}
	return a.items[len(a.items)-1]
}

func (a *AStack) Size() int {
	return len(a.items)
}

type IStack struct {
	items []int
}

func NewIStack() *IStack {
	return &IStack{[]int{}}
}

func (s *IStack) Push(x int) {
	s.items = append(s.items, x)
}

func (s *IStack) Pop() int {
	if len(s.items) == 0 {
		return -1
	}
	val := s.items[len(s.items)-1]
	s.items = s.items[:len(s.items)-1]
	return val
}

func (s *IStack) Peek() int {
	if len(s.items) == 0 {
		return -1
	}
	return s.items[len(s.items)-1]
}

func (s *IStack) Size() int {
	return len(s.items)
}

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

/*
3. Longest Substring Without Repeating Characters
*/

func lengthOfLongestSubstring(s string) int {
	start, maxLength := 0, 0
	usedChar := map[byte]int{}
	for i := 0; i < len(s); i++ {
		_, seen := usedChar[s[i]]
		if seen && start <= usedChar[s[i]] {
			start = usedChar[s[i]] + 1
		} else {
			len := i - start + 1
			if len > maxLength {
				maxLength = len
			}
		}
		usedChar[s[i]] = i
	}
	return maxLength
}

/*
28. Implement strStr()
*/

func strStr(haystack string, needle string) int {
	if len(needle) == 0 {
		return 0
	}
	for i := 0; i <= len(haystack)-len(needle); i++ {
		for j := 0; j < len(needle); j++ {
			if haystack[i+j] != needle[j] {
				break
			} else if j == len(needle)-1 {
				return i
			}
		}
	}
	return -1
}

/*
8. String to Integer (atoi)
*/

/*
questions:
what happens if a string has a non-numeric character? what should we return?
if it precedes the integral region, return 0
if it follows the integral region, ignore
multiple zeros?
what if the string causes integer overflow? - return int max, int min
negative values? "-00001"
whitespace characters?
*/

/*
first accepted
*/
func myAtoi(str string) int {
	str = strings.TrimLeftFunc(str, func(arg2 rune) bool {
		return unicode.IsSpace(arg2)
	})
	if len(str) == 0 {
		return 0
	}
	pos := true
	if str[0] == '-' {
		pos = false
	}
	if str[0] == '+' || str[0] == '-' {
		str = str[1:]
		if len(str) == 0 {
			return 0
		}
	}
	var rval int
	if hasPreIntegralRegion(str) {
		return 0
	}

	start, end := -1, len(str)-1

	for i, char := range str {
		isInt := isInt(char)
		if start == -1 && !isInt {
			return 0
		}
		if isInt && start == -1 {
			start = i
		}
		if !isInt {
			end = i - 1
			break
		}
	}

	mul := 1
	for i := end; i >= start; i-- {
		val := intFromChar(str[i]) * mul
		nrval := rval + val
		if nrval > 2147483647 {
			// integer overflow!
			if pos {
				return 2147483647
			}
			return -2147483648
		}
		rval = nrval
		mul *= 10
	}
	if !pos {
		rval *= -1
	}
	return rval
}

func intFromChar(c byte) int {
	return int(c - '0')
}

func isInt(c rune) bool {
	return (c >= '0' && c <= '9')
}

func hasPreIntegralRegion(str string) bool {
	for _, c := range str {
		if !isInt(c) {
			return true
		}
		break
	}
	return false
}

/*
refactored accepted
*/

func myAtoi(str string) int {
	sign, base, i := 1, 0, 0
	for i < len(str) && str[i] == ' ' {
		i++
	}
	if i < len(str) && (str[i] == '-' || str[i] == '+') {
		if str[i] == '-' {
			sign = -1
		}
		i++
	}
	for i < len(str) && (str[i] >= '0' && str[i] <= '9') {
		if base > math.MaxInt32/10 || (base == math.MaxInt32/10 && str[i]-'0' > 7) {
			if sign == 1 {
				return math.MaxInt32
			}
			return math.MinInt32
		}
		base = 10*base + int(str[i]-'0')
		i++
	}
	return base * sign
}

/*
206. Reverse Linked List
*/

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func reverseList(head *ListNode) *ListNode {
	if head == nil {
		return head
	}
	return reverseListHelper(head, nil)
}

func reverseListHelper(node *ListNode, p *ListNode) *ListNode {
	next := node.Next
	node.Next = p
	if next == nil {
		return node
	}
	return reverseListHelper(next, node)
}

/*
21. Merge Two Sorted Lists
*/

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	if l1 == nil {
		return l2
	}
	if l2 == nil {
		return l1
	}
	if l1.Val < l2.Val {
		l1.Next = mergeTwoLists(l1.Next, l2)
		return l1
	}
	l2.Next = mergeTwoLists(l1, l2.Next)
	return l2
}

/*
24. Swap Nodes in Pairs
*/

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func swapPairs(head *ListNode) *ListNode {
	if head == nil {
		return head
	}
	if head.Next == nil {
		return head
	}
	child := head.Next
	head.Next = swapPairs(head.Next.Next)
	child.Next = head
	return child
}

/*
104. Maximum Depth of Binary Tree
*/

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	maxLeft := maxDepth(root.Left)
	maxRight := maxDepth(root.Right)
	if maxLeft > maxRight {
		return maxLeft + 1
	}
	return maxRight + 1
}

/*
100. Same Tree
*/

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func isSameTree(p *TreeNode, q *TreeNode) bool {
	if p == nil && q == nil {
		return true
	}
	if p == nil || q == nil {
		return false
	}
	if p.Val == q.Val {
		return isSameTree(p.Left, q.Left) && isSameTree(p.Right, q.Right)
	}
	return false
}

/*
112. Path Sum
*/

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func hasPathSum(root *TreeNode, sum int) bool {
	if root == nil {
		return false
	}
	if root.Left == nil && root.Right == nil {
		return sum == root.Val
	}
	newSum := sum - root.Val
	return hasPathSum(root.Left, newSum) || hasPathSum(root.Right, newSum)
}

/*
257. Binary Tree Paths
*/

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */

/*
edge cases
empty root
1 child root
*/

func binaryTreePaths(root *TreeNode) []string {
	if root == nil {
		return []string{}
	}
	strVal := strconv.Itoa(root.Val)
	if root.Left == nil && root.Right == nil {
		return []string{strVal}
	}
	mergedPaths := append(binaryTreePaths(root.Left), binaryTreePaths(root.Right)...)
	paths := []string{}
	for _, path := range mergedPaths {
		paths = append(paths, strVal+"->"+path)
	}
	return paths
}

/*
98. Validate Binary Search Tree
*/

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */

/*
edge cases:
empty root
one child root
*/

func isValidBST(root *TreeNode) bool {
	return isValidBSTHelper(root, math.MinInt64, math.MaxInt64)
}

func isValidBSTHelper(root *TreeNode, min, max int) bool {
	if root == nil {
		return true
	}
	if root.Val <= min || root.Val >= max {
		return false
	}
	return isValidBSTHelper(root.Left, min, root.Val) &&
		isValidBSTHelper(root.Right, root.Val, max)
}

/*
225. Implement Stack using Queues
*/
type MyStack struct {
	q *Queue
}

/** Initialize your data structure here. */
func Constructor() MyStack {
	return MyStack{q: NewQueue()}
}

/** Push element x onto stack. */
func (this *MyStack) Push(x int) {
	this.q.Push(x)
	for i := 0; i < this.q.Size()-1; i++ {
		this.q.Push(this.q.Peek())
		this.q.Pop()
	}
}

/** Removes the element on top of the stack and returns that element. */
func (this *MyStack) Pop() int {
	return this.q.Pop()
}

/** Get the top element. */
func (this *MyStack) Top() int {
	return this.q.Peek()
}

/** Returns whether the stack is empty. */
func (this *MyStack) Empty() bool {
	return this.q.IsEmpty()
}

/**
 * Your MyStack object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Push(x);
 * param_2 := obj.Pop();
 * param_3 := obj.Top();
 * param_4 := obj.Empty();
 */

/*
20. Valid Parentheses
*/

func isValid(s string) bool {
	stack := NewAStack()
	for _, c := range s {
		sc := string(c)
		if strings.Contains("({[", sc) {
			stack.Push(sc)
		} else {
			if stack.Peek() != matchingOpen(sc) {
				return false
			}
			stack.Pop()
		}
	}
	return stack.Size() == 0
}

func matchingOpen(s string) string {
	if s == ")" {
		return "("
	} else if s == "}" {
		return "{"
	} else {
		return "["
	}
}

/*
232. Implement Queue using Stacks
*/

type MyQueue struct {
	input  *IStack
	output *IStack
}

/** Initialize your data structure here. */
func Constructor() MyQueue {
	return MyQueue{NewIStack(), NewIStack()}
}

/** Push element x to the back of queue. */
func (this *MyQueue) Push(x int) {
	this.input.Push(x)
}

/** Removes the element from in front of queue and returns that element. */
func (this *MyQueue) Pop() int {
	this.Peek()
	return this.output.Pop()
}

/** Get the front element. */
func (this *MyQueue) Peek() int {
	if this.output.Size() == 0 {
		for this.input.Size() > 0 {
			this.output.Push(this.input.Pop())
		}
	}
	return this.output.Peek()
}

/** Returns whether the queue is empty. */
func (this *MyQueue) Empty() bool {
	return this.input.Size() == 0 && this.output.Size() == 0
}

/**
 * Your MyQueue object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Push(x);
 * param_2 := obj.Pop();
 * param_3 := obj.Peek();
 * param_4 := obj.Empty();
 */

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

/*
7. Reverse Integer
*/

func reverse(x int) int {
	result := 0
	for x != 0 {
		tail := x % 10
		nResult := result*10 + tail
		if nResult > math.MaxInt32 || nResult < math.MinInt32 {
			return 0
		}
		result = nResult
		x /= 10
	}
	return result
}

/*
231. Power of Two
*/

func isPowerOfTwo(n int) bool {
	if n < 0 {
		return false
	}
	count := 0
	var i uint
	for i = 0; i < 32; i++ {
		if ((1 << i) & n) != 0 {
			count++
		}
	}
	return count == 1
}
