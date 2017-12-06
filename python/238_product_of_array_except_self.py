'''
238. Product of Array Except Self
'''

class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        '''
        input:
        [2, 3, 5]
        output:
        [3 * 5, 2 * 5, 2 * 3]
        n * n = O(n^2) brute force implementation
        can we do O(n log n)? will sorting help us? doesn't seem like it
        2, 2 * 3
        5, 5 * 3
        running_product[i] = product of elements in 0 ... i
        reverse_running_product[i] = ""
        '''
        A = [0, nums[0]]
        for i, x in enumerate(nums[1:]):
            running_product.append(running_product[i] * x)
        # now running product is populated
        reverse_running_product = nums[-1]
        result = [running_product[-2]]
        for i, x in enumerate(reversed(nums[:-1])):
            result.append(reverse_running_product + running_product[-i])
            reverse_running_product = x * reverse_running_product
        return reversed(result)

        '''
        [2, 3, 5]
        running_product = [2]
        i = 0
        x = 3
        running_product = [2, 2*3]
        i = 1
        x = 5
        running_product = [2, 6, 6*5]

        reverse_running_product = 5
        result = []

        [2, 3] reversed = [3, 2]
        i = 0
        x = 3
        result = []
        '''

        result = [1]
        for i in range(1, len(nums)):
            result.append(nums[i-1] * result[-1])
        running_product = nums[-1]
        for i in reversed(range(len(nums)-1)):
            result[i] *= running_product
            running_product *= nums[i]
        return result

        '''
        result = [1]
        result = [1, 2 * 1]
        result = [1, 2, 3 * 2]
        result = [1, 2, 6]
        prod = 5
        0, 1 = 1, 0
        result = [1, 2, 6]
        result[1] *= 5
        prod = 5 * 3 = 15
        result = [1, 2*5, 6]
        result = [1, 2*5, 2*3]
        result[0] = 1 * 5 * 3 = 15
        result = [3 * 5, 2 * 5, 2 * 3]
        '''

