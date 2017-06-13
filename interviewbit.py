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

'''
Merge Intervals
'''

# Definition for an interval.
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

# @param intervals, a list of Intervals
# @param new_interval, a Interval
# @return a list of Interval
def insert(self, intervals, new_interval):
    ret = []
    blob = new_interval
    for i, interval in enumerate(intervals):
        if blob == None:
            ret.append(interval)
            continue
        if self.overlaps(interval, blob):
            blob = Interval(min(interval.start, blob.start), max(interval.end, blob.end))
        else:
            if blob.end < interval.start:
                ret.append(blob)
                blob = None
            ret.append(interval)
    if blob != None:
        ret.append(blob)
    return ret

def overlaps(self, x, y):
    if x is None or y is None:
        return False
    if x.start > y.start:
        x, y = y, x
    return x.end > y.start

'''
Add One To Number
'''

# @param A : list of integers
# @return a list of integers
def plusOne(self, A):
    overflow = False
    for i in reversed(xrange(len(A))):
        if A[i] < 9:
            A[i] += 1
            overflow = False
            break
        else:
            A[i] = 0
            overflow = True
    if overflow:
        temp = A
        A = [1]
        A.extend(temp)
    return self.stripLeadingZeroes(A)

def stripLeadingZeroes(self, A):
    j = 0
    for i in xrange(len(A)):
        if A[i] != 0:
            j = i
            break
    return A[j:]

'''
Largest Number

Given a list of non negative integers, arrange them such that they
form the largest number.

For example:
Given [3, 30, 34, 5, 9], the largest formed number is 9534330.
'''

# @param A : tuple of integers
# @return a strings
def largestNumber(self, A):
    if len(A) == 0:
        return "0"
    temp = sorted(A, cmp=self.numCmp)
    if temp[0] == 0:
        return "0"
    return "".join(str(x) for x in temp)

def numCmp(self, x, y):
    xstr, ystr = str(x), str(y)
    if int(xstr + ystr) > int(ystr + xstr):
        return -1
    else:
        return 1
