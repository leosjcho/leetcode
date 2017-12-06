'''
453. Minimum Moves to Equal Array Elements
'''

class Solution(object):
    def minMoves(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        # first approach
        '''
        minval = float('inf')
        for x in nums:
            minval = min(minval, x)
        moves = 0
        for x in nums:
            moves += x - minval
        return moves
        '''

        # after seeing one liner top solution
        '''
        minval = min(nums)
        numsum = sum(nums)
        targetsum = len(nums) * minval
        return numsum - targetsum
        '''

        # one liner top solution
        return sum(nums) - len(nums) * min(nums)


