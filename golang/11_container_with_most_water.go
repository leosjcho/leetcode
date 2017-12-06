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

