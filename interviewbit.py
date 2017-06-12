'''
Arrays
'''

# @param X : list of integers
# @param Y : list of integers
# Points are represented by (X[i], Y[i])
# @return an integer
def coverPoints(self, X, Y):
    if len(X) < 2:
        return 0
    steps = 0
    for i in xrange(len(X)-1):
        dx = abs(X[i+1] - X[i])
        dy = abs(Y[i+1] - Y[i])
        steps += min(dx, dy)
        steps += max(dx, dy) - min(dx, dy)
    return steps

'''
Noble Integer

Given an integer array, find if an integer p exists in the array such
that the number of integers greater than p in the array equals to p
If such an integer is found return 1 else return -1.

# @param A : list of integers
# @return an integer

naive implementation:
    for each element, compare it against every other element and keep
    count
    O(n^2) time
better implementaiton:
    sort integers
    iterate over sorted list
        if # of elements after element == element value, return true
    return false
'''

def solve(self, A):
    A.sort()
    for i in xrange(len(A)):
        # strictly larger elements at higher indices
        if i < len(A)-1 and A[i] == A[i+1]:
            continue
        if len(A) - i - 1 == A[i]:
            return 1
    return -1

'''
Wave Array

Given an array of integers, sort the array into a wave like array
and return it,
In other words, arrange the elements into a sequence such that a1
>= a2 <= a3 >= a4 <= a5.....
'''

# @param A : list of integers
# @return a list of integers
def wave(self, A):
    A.sort()
    i = 0
    while i < len(A)-1:
        A[i], A[i+1] = A[i+1], A[i]
        i += 2
    return A
