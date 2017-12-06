'''
54. Spiral Matrix
'''

class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        shifts = ((0, 1), (1, 0), (0, -1), (-1, 0))
        direction = x = y = 0
        spiral = []
        m = len(matrix)
        if m == 0:
            return []
        n = len(matrix[0])

        for _  in range(m*n):
            spiral.append(matrix[x][y])
            matrix[x][y] = float('-inf')
            dx, dy = shifts[direction][0], shifts[direction][1]
            nx, ny = x + dx, y + dy
            if nx >= m or ny >= n or matrix[nx][ny] == float('-inf'):
                direction = (direction + 1) % 4
                dx, dy = shifts[direction][0], shifts[direction][1]
                nx, ny = x + dx, y + dy
            x, y = nx, ny
        return spiral

