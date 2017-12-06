'''
200. Number of Islands
'''

import collections

class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        WATER, LAND = map(str, range(2))
        Coord = collections.namedtuple("Coord", ("x", "y"))

        def mark_island_dfs(loc, visited):
            if loc.x < 0 or loc.x >= len(grid) \
                or loc.y < 0 or loc.y >= len(grid[0]):
                return
            if grid[loc.x][loc.y] == LAND and loc not in visited:
                visited.add(loc)
                for d in (0,1), (1,0), (-1,0), (0,-1):
                    new_x, new_y = loc.x + d[0], loc.y + d[1]
                    mark_island_dfs(Coord(new_x, new_y), visited)

        visited = set()
        island_count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                loc = Coord(i, j)
                if loc not in visited and grid[i][j] == LAND:
                    island_count += 1
                    mark_island_dfs(loc, visited)
                    visited.add(loc)
        return island_count

        '''
        concise soln
        '''

        def sink(i, j):
            if 0 <= i < len(grid) and 0 <= j < len(grid[0]) \
                and grid[i][j] == '1':
                    grid[i][j] = '0'
                    map(sink, (i+1, i-1, i, i), (j, j, j+1, j-1))
                    return 1
            return 0
        return sum(sink(i, j) for i in range(len(grid)) \
                   for j in range(len(grid[0])))

