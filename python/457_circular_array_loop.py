'''
457. Circular Array Loop
'''
class Solution(object):

    # 0 1 2
    # -1
    # 2
    # 3 - 1
    # -3
    # 3 % 4 = 3
    # 3 - 0 = 3
    # 4 % 4 = 0
    # 3- 0
    def nextIndex(self, n, old, incr):
        new_val = old + incr
        if new_val < 0:
            return n - 1 - (abs(new_val) - 1 % n)
        else:
            return new_val % n

    def circularArrayLoop(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """

        for i in range(len(nums)):
            # if the value has already been set to 0, skip
            if nums[i] == 0:
                continue
            else:
                forward = False
                backward = False
                next_index = self.nextIndex(len(nums), i, nums[i])
                indexes_in_loop = 1
                while nums[next_index] != 0:
                    index_inc = nums[next_index]
                    nums[next_index] = 0
                    next_index = self.nextIndex(len(nums), next_index,
                        index_inc)
                    indexes_in_loop += 1
                if indexes_in_loop > 1 and forward != backward:
                    return True
        return False


