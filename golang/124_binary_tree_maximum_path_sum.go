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

