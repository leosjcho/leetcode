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

