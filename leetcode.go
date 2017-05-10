package main

import (
	"container/heap"
	"math"
	"strconv"
	"strings"
)

func main() {

}

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

/*
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
*/

// explcit
func productExceptSelf(nums []int) []int {
	results := make([]int, len(nums))
	runningTotal := 1
	for i := range nums {
		results[i] = runningTotal
		runningTotal *= nums[i]
	}
	runningTotal = 1
	for i := len(nums) - 1; i >= 0; i-- {
		results[i] *= runningTotal
		runningTotal *= nums[i]
	}
	return results
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
/*
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
*/

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
		if base > math.MaxInt32/10 || (base == math.MaxInt32/10 &&
			str[i]-'0' > 7) {
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

type ListNode struct {
	Val  int
	Next *ListNode
}

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

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

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
	mergedPaths := append(binaryTreePaths(root.Left),
		binaryTreePaths(root.Right)...)
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
// func Constructor() MyQueue {
// 	return MyQueue{NewIStack(), NewIStack()}
// }

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

/*
367. Valid Perfect Square
*/

func isPerfectSquare(num int) bool {
	lo, hi := 0, math.MaxUint32
	for lo <= hi {
		mid := (hi-lo)/2 + lo
		if num == mid*mid {
			return true
		}
		if num < mid*mid {
			hi = mid - 1
		} else {
			lo = mid + 1
		}
	}
	return false
}

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

/*
292. Nim Game
*/

/*
if n == 1, 2, or 3, then we win!
if n == 4, what happens? we lose!
if n == 5? we win!
if n == 6? we win!
if n == 7? we win!
if n == 8? we lose!
etc...
*/

func canWinNim(n int) bool {
	return n%4 != 0
}

/*
344. Reverse String
*/

/*
empty string:
i = 0, j = 0-1
i is not less than j. returns empty string
"a"
i = 0, j = 1-1 = 0
i is not less than j
returns "a"
"ab"
i = 0, j = 2-1 = 1
i is less than j.
swaps i and j
i = 1, j = 0
i is not less than j
returns "ba"
*/

func reverseString(s string) string {
	i, j := 0, len(s)-1
	r := []rune(s)
	for i < j {
		r[i], r[j] = r[j], r[i]
		i++
		j--
	}
	return string(r)
}

/*
2. Add Two Numbers
*/

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */

// while l1 && l2 exist
// sum = l1 and l2 and carryover
// carryover = sum // 10 (integer division)
// sum %= 10
// add new node to sumlist with sum
// while l1 exists
// sum = l1 + carryover
// perform same operation as above
// while l2 exists
// perform same operations as above
// if carryover > 0
// add new node to sumlist with carryover

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	carryover := 0
	sumHead := &ListNode{}
	sumTail := sumHead
	for l1 != nil || l2 != nil || carryover > 0 {
		l1v, l2v := 0, 0
		if l1 != nil {
			l1v = l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			l2v = l2.Val
			l2 = l2.Next
		}
		sum := l1v + l2v + carryover
		n := &ListNode{sum % 10, nil}
		carryover = sum / 10
		sumTail.Next = n
		sumTail = n
	}
	return sumHead.Next
}

/*
l1 = [2]
l2 = [9, 5]
carryover = 0
sumhead = empty node
sumtail = sumhead
l1 and l2 not nil
sum = 2 + 9 + 0 = 11
c = 11 / 10 = 1
sum = 1
sumtail.next = {1, nil}
carryover = 1, sumTail = {1, nil}
l1 = nil, l2 = 5
sum = 0 + 5 + 1 = 6
c = 6 / 10 = 0
node = {6, nil}
output:
1 -> 6
*/

/*
371. Sum of Two Integers
*/

func getSum(a int, b int) int {
	if b == 0 {
		return a
	}
	return getSum(a^b, (a&b)<<1)
}

/*
146. LRU Cache
*/

type LRUCache struct {
	data     map[int]int
	lives    map[int]int
	purgeQ   []int // queue
	count    int
	capacity int
}

// func Constructor(capacity int) LRUCache {
// 	return LRUCache{
// 		data:     map[int]int{},
// 		lives:    map[int]int{},
// 		purgeQ:   []int{},
// 		count:    0,
// 		capacity: capacity,
// 	}
// }

// updates timestamp
// never evicts elements
func (this *LRUCache) Get(key int) int {
	v, ok := this.data[key]
	if !ok {
		return -1
	}
	if v != -1 {
		this.lives[key]++
		this.purgeQ = append(this.purgeQ, key)
	}
	return v
}

// update timestamp
// insert key
// evict if at capacity
func (this *LRUCache) Put(key int, value int) {
	v, ok := this.data[key]
	// is this a new item?
	if !ok || v == -1 {
		this.count++
	}

	this.lives[key]++
	this.purgeQ = append(this.purgeQ, key)
	this.data[key] = value

	for this.count > this.capacity {
		candidate := this.purgeQ[0]
		this.purgeQ = this.purgeQ[1:]
		this.lives[candidate]--
		if this.lives[candidate] == 0 {
			// evict!
			this.data[candidate] = -1
			this.count--

		}
	}
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * obj := Constructor(capacity);
 * param_1 := obj.Get(key);
 * obj.Put(key,value);
 */

/*
169. Majority Element
*/

// O(n) time complexity, O(n) space complexity
func majorityElement(nums []int) int {
	counts := map[int]int{}
	maxElement, maxCount := 0, 0
	for _, num := range nums {
		counts[num]++
		if counts[num] > maxCount {
			maxElement = num
			maxCount = counts[num]
		}
	}
	return maxElement
}

/*
13. Roman to Integer
*/

func romanToInt(s string) int {
	values := map[byte]int{
		'I': 1,
		'V': 5,
		'X': 10,
		'L': 50,
		'C': 100,
		'D': 500,
		'M': 1000,
	}
	result := values[s[len(s)-1]]
	for i := len(s) - 2; i >= 0; i-- {
		v := values[s[i]]
		if v < values[s[i+1]] {
			result -= v
		} else {
			result += v
		}
	}
	return result
}

/*
4. Median of Two Sorted Arrays
*/

// https://discuss.leetcode.com/topic/28602/concise-java-solution-based-on-
// binary-search

func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	m, n := len(nums1), len(nums2)
	l, r := (m+n+1)/2, (m+n+2)/2
	return (getKth(nums1, nums2, 0, 0, l) + getKth(nums1, nums2, 0, 0, r)) / 2
}

func getKth(nums1, nums2 []int, aStart, bStart, k int) float64 {
	if aStart > len(nums1)-1 {
		return float64(nums2[bStart+k-1])
	}
	if bStart > len(nums2)-1 {
		return float64(nums1[aStart+k-1])
	}
	if k == 1 {
		return float64(min(nums1[aStart], nums2[bStart]))
	}
	aMid, bMid := math.MaxInt32, math.MaxInt32
	if aStart+k/2-1 < len(nums1) {
		aMid = nums1[aStart+k/2-1]
	}
	if bStart+k/2-1 < len(nums2) {
		bMid = nums2[bStart+k/2-1]
	}
	if aMid < bMid {
		return getKth(nums1, nums2, aStart+k/2, bStart, k-k/2)
	} else {
		return getKth(nums1, nums2, aStart, bStart+k/2, k-k/2)
	}
}

/*
266. Palindrome Permutation
*/

func canPermutePalindrome(s string) bool {
	seen := map[rune]bool{}
	for _, c := range s {
		_, ok := seen[c]
		if ok {
			delete(seen, c)
		} else {
			seen[c] = true
		}
	}
	if len(s)%2 == 0 {
		return len(seen) == 0
	}
	return len(seen) == 1
}

/*
155. Min Stack
*/

type MinStack struct {
	items []int
	mins  []int
}

/** initialize your data structure here. */
// func Constructor() MinStack {
// 	return MinStack{
// 		items: []int{},
// 		mins:  []int{},
// 	}
// }

func (this *MinStack) Push(x int) {
	m := len(this.mins)
	if m == 0 {
		this.mins = append(this.mins, x)
	} else if x <= this.mins[m-1] {
		this.mins = append(this.mins, x)
	}
	this.items = append(this.items, x)
}

func (this *MinStack) Top() int {
	return this.items[len(this.items)-1]
}

func (this *MinStack) Pop() {
	n := len(this.items)
	m := len(this.mins)
	item := this.items[n-1]
	// pop from items stack
	this.items = this.items[:n-1]
	// if the item was a min value
	if this.mins[m-1] == item {
		// pop from mins stack
		this.mins = this.mins[:m-1]
	}
}

func (this *MinStack) GetMin() int {
	return this.mins[len(this.mins)-1]
}

/**
 * Your MinStack object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Push(x);
 * obj.Pop();
 * param_3 := obj.Top();
 * param_4 := obj.GetMin();
 */

/*
5. Longest Palindromic Substring
*/

func longestPalindrome(s string) string {
	longest := 0
	ll, lr := 0, 0
	for i := 0; i < len(s); i++ {
		singleCount, sl, sr := palindromicSize(s, i, i)
		dblCount, dl, dr := palindromicSize(s, i, i+1)
		if singleCount > longest {
			longest = singleCount
			ll, lr = sl, sr
		}
		if dblCount > longest {
			longest = dblCount
			ll, lr = dl, dr
		}
	}
	return s[ll : lr+1]
}

func palindromicSize(s string, i, j int) (int, int, int) {
	if j >= len(s) || s[i] != s[j] {
		return 0, i, j
	}
	for i >= 0 && j < len(s) && s[i] == s[j] {
		i--
		j++
	}
	return j - i - 1, i + 1, j - 1
}

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

/*
148. Sort List
*/

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func sortList(head *ListNode) *ListNode {
	// get "head" of right half
	if head == nil || head.Next == nil {
		return head
	}
	rightHalfHead := getRightHalfHead(head)
	leftHead := sortList(head)
	rightHead := sortList(rightHalfHead)
	return merge(leftHead, rightHead)
}

func getRightHalfHead(head *ListNode) *ListNode {
	var stepperParent *ListNode
	stepper, jumper := head, head
	for jumper != nil && jumper.Next != nil {
		stepperParent = stepper
		stepper = stepper.Next
		jumper = jumper.Next.Next
	}
	stepperParent.Next = nil
	return stepper
}

func merge(left *ListNode, right *ListNode) *ListNode {
	l := &ListNode{}
	p := l
	for left != nil && right != nil {
		if left.Val < right.Val {
			p.Next = left
			left = left.Next
		} else {
			p.Next = right
			right = right.Next
		}
		p = p.Next
	}
	if left != nil {
		p.Next = left
	}
	if right != nil {
		p.Next = right
	}
	return l.Next
}

/*
66. Plus One
*/

func plusOne(digits []int) []int {
	n := len(digits)
	for i := n - 1; i >= 0; i-- {
		if digits[i] < 9 {
			digits[i]++
			return digits
		} else {
			digits[i] = 0
		}
	}
	digits = append([]int{1}, digits...)
	return digits
}

/*
70. Climbing Stairs
*/

func climbStairs(n int) int {
	if n < 3 {
		return n
	}
	dp := make([]int, n+1)
	dp[1] = 1
	dp[2] = 2
	for i := 3; i <= n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}

/*
121. Best Time to Buy and Sell Stock
*/

func maxProfit(prices []int) int {
	maxProfit := 0
	minSeen := math.MaxInt32
	for _, p := range prices {
		maxProfit = max(maxProfit, p-minSeen)
		minSeen = min(minSeen, p)
	}
	return maxProfit
}

/*
198. House Robber
*/

func rob(nums []int) int {
	prev, curr := 0, 0
	for _, p := range nums {
		temp := curr
		curr = max(prev+p, curr)
		prev = temp
	}
	return curr
}

/*
200. Number of Islands
*/

func numIslands(grid [][]byte) int {
	numIslandCount := 0
	visited := make([][]bool, len(grid))
	for i := 0; i < len(grid); i++ {
		visited[i] = make([]bool, len(grid[i]))
	}
	// for each tile
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			// if it's a water tile, mark it as visited, continue
			if grid[i][j] == '0' {
				continue
			} else {
				// if it's a land tile
				// if it's unvisited
				if !visited[i][j] {
					numIslandCount++
					visitAdjacentIslands(i, j, visited, grid)
				}
			}
		}
	}
	return numIslandCount
}

func visitAdjacentIslands(i, j int, visited [][]bool, grid [][]byte) {
	if i < 0 || j < 0 || i >= len(grid) || j >= len(grid[0]) {
		return
	}
	if grid[i][j] == '0' || visited[i][j] == true {
		return
	}
	visited[i][j] = true
	visitAdjacentIslands(i-1, j, visited, grid)
	visitAdjacentIslands(i+1, j, visited, grid)
	visitAdjacentIslands(i, j-1, visited, grid)
	visitAdjacentIslands(i, j+1, visited, grid)
}

/*
110. Balanced Binary Tree
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
***** Incorrect solution from misinterpreted solution

type NodeWrapper struct {
	Node  *TreeNode
	Depth int
}

func isBalanced(root *TreeNode) bool {
	// visit every node, updating min and max height seen thus far
	// check if difference > 1
	minDepth, maxDepth := math.MaxInt32, 0
	stack := []*NodeWrapper{}
	stack = append(stack, &NodeWrapper{root, 0})
	for len(stack) > 0 {
		n := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if n.Node == nil {
			continue
		}
		if n.Node.Left == nil || n.Node.Right == nil {
			minDepth = min(minDepth, n.Depth)
			maxDepth = max(maxDepth, n.Depth)
		}
		stack = append(stack, &NodeWrapper{n.Node.Left, n.Depth + 1})
		stack = append(stack, &NodeWrapper{n.Node.Right, n.Depth + 1})
	}
	fmt.Println(maxDepth, minDepth)
	return maxDepth-minDepth < 2
}
*/

func isBalanced(root *TreeNode) bool {
	return dfsHeight(root) != -1
}

func dfsHeight(node *TreeNode) int {
	if node == nil {
		return 0
	}
	leftHeight := dfsHeight(node.Left)
	if leftHeight == -1 {
		return -1
	}
	rightHeight := dfsHeight(node.Right)
	if rightHeight == -1 {
		return -1
	}
	diff := rightHeight - leftHeight
	if diff < 0 {
		diff = -diff
	}
	if diff > 1 {
		return -1
	}
	return max(leftHeight, rightHeight) + 1
}

/*
108. Convert Sorted Array to Binary Search Tree
*/

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */

func sortedArrayToBST(nums []int) *TreeNode {
	if len(nums) == 0 {
		return nil
	}
	mid := (len(nums) - 1) / 2
	n := &TreeNode{Val: nums[mid]}
	n.Left = sortedArrayToBST(nums[0:mid])
	n.Right = sortedArrayToBST(nums[mid+1 : len(nums)])
	return n
}

/*
124. Binary Tree Maximum Path Sum
*/

/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func maxPathSum(root *TreeNode) int {
	mp := math.MinInt32
	maxPathSumHelper(root, &mp)
	return mp
}

func maxPathSumHelper(root *TreeNode, mp *int) int {
	if root == nil {
		return 0
	}
	leftMax := max(0, maxPathSumHelper(root.Left, mp))
	rightMax := max(0, maxPathSumHelper(root.Right, mp))
	*mp = max(*mp, leftMax+rightMax+root.Val)
	return max(leftMax, rightMax) + root.Val
}

/*
23. Merge k Sorted Lists
*/

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */

type PQItem struct {
	node  *ListNode
	index int
}

type PriorityQueue []*PQItem

func (pq PriorityQueue) Len() int {
	return len(pq)
}

func (pq PriorityQueue) Less(i, j int) bool {
	return pq[i].node.Val < pq[j].node.Val
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}

func (pq *PriorityQueue) Push(x interface{}) {
	*pq = append(*pq, x.(*PQItem))
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	item := old[len(*pq)-1]
	*pq = old[0 : len(*pq)-1]
	return item
}

func heapify(lists []*ListNode) *PriorityQueue {
	pq := make(PriorityQueue, 0)
	for i := range lists {
		if lists[i] != nil {
			pq = append(pq, &PQItem{lists[i], i})
		}
	}
	heap.Init(&pq)
	return &pq
}

func mergeKLists(lists []*ListNode) *ListNode {
	// build heap with list heads
	pq := heapify(lists) // O(n log n)
	curr := &ListNode{}
	headPointer := &ListNode{Next: curr} // pointer to head
	// while heap is not empty
	for pq.Len() > 0 {
		// get node and it's index in the list of lists
		item := heap.Pop(pq).(*PQItem) // O(log n)
		node, i := item.node, item.index
		lists[i] = node.Next
		if lists[i] != nil {
			newItem := &PQItem{lists[i], i}
			heap.Push(pq, newItem)
		}
		curr.Next = node
		curr = node
	}
	return headPointer.Next.Next
}

// overall runtime O(n log n)
// space complexity O(n)

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

/*
347. Top K Frequent Elements
*/

/*
Given a non-empty array of integers, return the k most frequent elements.

For example,
Given [1,1,1,2,2,3] and k = 2, return [1,2].
*/

// count occurances of each number in hash table
// then build max heap with (num, occurances) tuples

func topKFrequent(nums []int, k int) []int {
	occurances := countOccurances(nums) // returns map[int]int
	// build max heap with occurances
	pq := NewHeap(occurances)
	top := []int{}
	for i := 0; i < k; i++ {
		top = append(top, heap.Pop(pq).(*KFreqItem).num)
	}
	return top
}

func countOccurances(nums []int) map[int]int {
	counts := map[int]int{}
	for _, v := range nums {
		counts[v]++
	}
	return counts
}

type KFreqItem struct {
	num   int
	count int
}

type KFreqPQ []*KFreqItem

func (pq KFreqPQ) Len() int {
	return len(pq)
}

func (pq KFreqPQ) Less(i, j int) bool {
	return pq[i].count > pq[j].count
}

func (pq KFreqPQ) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}

func (pq *KFreqPQ) Push(x interface{}) {
	*pq = append(*pq, x.(*KFreqItem))
}

func (pq *KFreqPQ) Pop() interface{} {
	n := len(*pq)
	old := *pq
	item := old[n-1]
	*pq = old[:n-1]
	return item
}

func NewHeap(occurances map[int]int) *KFreqPQ {
	pq := make(KFreqPQ, 0)
	for k, v := range occurances {
		pq = append(pq, &KFreqItem{k, v})
	}
	heap.Init(&pq)
	return &pq
}

/*
53. Maximum Subarray
*/

/*
Find the contiguous subarray within an array (containing at least one number)
which has the largest sum.

For example, given the array [-2,1,-3,4,-1,2,1,-5,4],
the contiguous subarray [4,-1,2,1] has the largest sum = 6.
*/

/*
// -2 1 -3 4
// gm = -inf lm = -inf
-2:
lm = -2
gm = -2
1:
lm = max(-2 + 1, 1) = 1
gm = max(-2, 1) = 1
-3:
lm = max(1 - 3, -3) = -2
gm = max(1, -2) = 1
4:
lm = max(-2+4, 4) = 4
gm = max(1, 4) = 4
return 4
*/

func maxSubArray(nums []int) int {
	globalMax, localMax := math.MinInt32, math.MinInt32
	for _, x := range nums {
		localMax = max(localMax+x, x)
		globalMax = max(globalMax, localMax)
	}
	return globalMax
}

/*
6. ZigZag Conversion
*/

func convert(s string, numRows int) string {
	if numRows < 2 {
		return s
	}
	rows := generateRows(s, numRows) // [][]byte
	zigZagged := []byte{}
	for i := range rows {
		zigZagged = append(zigZagged, rows[i]...)
	}
	return string(zigZagged)
}

func generateRows(s string, numRows int) [][]byte {
	rows := init2DByteSlice(numRows, 0)
	i, n := 0, len(s)
	for i < n {
		scan(rows, s, &i, true)
		scan(rows, s, &i, false)
	}
	return rows
}

func scan(rows [][]byte, s string, i *int, down bool) {
	for j := 0; j < len(rows)-1; j++ {
		var x int
		if down {
			x = j
		} else {
			x = len(rows) - 1 - j
		}
		if *i == len(s) {
			return
		}
		rows[x] = append(rows[x], s[*i])
		*i++
	}
}

func init2DByteSlice(rows, cols int) [][]byte {
	arr := make([][]byte, rows)
	for i := range arr {
		arr[i] = make([]byte, cols)
	}
	return arr
}

/*
258. Add Digits
*/

/*
Given a non-negative integer num, repeatedly add all its digits until the
result has only one digit.

For example:

Given num = 38, the process is like: 3 + 8 = 11, 1 + 1 = 2. Since 2 has only
one digit, return it.

Follow up:
Could you do it without any loop/recursion in O(1) runtime?
*/

func addDigits(num int) int {
	return 1 + (num-1)%9
}

/*
338. Counting Bits
*/

/*
Given a non negative integer number num. For every numbers i in the range
0 ≤ i ≤ num calculate the number of 1's in their binary representation and
return them as an array.

Example:
For num = 5 you should return [0,1,1,2,1,2].

0000 0
0001 1
0010 1
0011 2
0100 1
0101 2
0110 2
0111 3
1000         1
1001 1 + 1 = 2
1010 1 + 1 = 2
1011 1 + 2 = 3
1100 1 + 1 = 2
1101 1 + 2 = 3
1110 1 + 2 = 3
1111 1 + 3 = 4
*/

func countBits(num int) []int {
	dp := make([]int, num+1)
	for i := 1; i <= num; i++ {
		dp[i] = dp[i>>1] + (i & 1)
	}
	return dp
}
