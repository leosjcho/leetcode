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

