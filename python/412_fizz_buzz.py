'''
412. Fizz Buzz
'''

class Solution(object):

    def fizzBuzz(self, n):
        """
        :type n: int
        :rtype: List[str]
        """

        # first attempt
        '''
        rlist = []
        for i in range(1, n + 1):
            x = str(i)
            if i % 3 == 0 and i % 5 == 0:
                x = "FizzBuzz"
            elif i % 3 == 0:
                x = "Fizz"
            elif i % 5 == 0:
                x = "Buzz"
            rlist.append(x)
        return rlist
        '''

        # one liner top solution
        '''
        return ['Fizz' * (not i % 3) + 'Buzz' * (not i % 5) or str(i)
        for i in range(1, n+1)]
        '''

        # attempt to speed up runtime by removing usage of mod ('%'), but
        # no faster than the above one liner lol
        # though still faster than original solution
        rlist = []
        f = 0
        b = 0
        for i in range(1, n + 1):
            x = str(i)
            f += 1
            b += 1
            if f == 3 and b == 5:
                x = "FizzBuzz"
                f = 0
                b = 0
            elif f == 3:
                x = "Fizz"
                f = 0
            elif b == 5:
                x = "Buzz"
                b = 0
            rlist.append(x)
        return rlist


