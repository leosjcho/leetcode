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

