'''
37. Sudoku Solver
'''

class Solution(object):
    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        rows = len(board)
        cols = len(board[0])

        def isSolutionValid():
            for i in range(rows):
                if hasDuplicate(board[i]):
                    return False
            for j in range(cols):
                if hasDuplicate(columnElements(j)):
                    return False
            for i in range(0, rows, 3):
                for j in range(0, rows, 3):
                    if hasDuplicateInSquare(i, j):
                        return False
            return True

        def columnElements(j):
            return [row[j] for row in board]

        def hasDuplicate(elements):
            seen = {}
            for e in elements:
                if e == ".":
                    continue
                if e in seen:
                    return True
                else:
                    seen[e] = True
            return False

        def hasDuplicateInSquare(i,j):
            rowSeg = i // 3
            colSeg = j // 3
            squareElements = []
            for i in range(rowSeg * 3, (rowSeg+1) * 3):
                for j in range(colSeg*3, (colSeg+1) * 3):
                    squareElements.append(board[i][j])
            return hasDuplicate(squareElements)

        def isValid(i, j):
            return not hasDuplicate(board[i]) and not hasDuplicate(columnElements(j)) and not hasDuplicateInSquare(i, j)

        def backtrack(i, j, emptyCount):
            def nextIndices(i, j):
                if j < cols - 1:
                    return i, j+1
                else:
                    return i+1, 0

            if emptyCount == 0:
                return isSolutionValid()
            if board[i][j] == ".":
                for x in range(1, 10):
                    board[i][j] = str(x)
                    if isValid(i,j):
                        ii, jj = nextIndices(i, j)
                        if backtrack(ii, jj, emptyCount-1):
                            return True
                board[i][j] = "."
            else:
                ii, jj = nextIndices(i, j)
                if backtrack(ii, jj, emptyCount):
                    return True

        def countEmptySpots():
            count = 0
            for i in range(rows):
                for j in range(cols):
                    if board[i][j] == ".":
                        count += 1
            return count

        emptyCount = countEmptySpots()
        backtrack(0, 0, emptyCount)

