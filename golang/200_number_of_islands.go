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

