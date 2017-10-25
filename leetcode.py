'''
388. Longest Absolute File Path
'''

# first naive approach
class Solution(object):

    def normalize_string(self, input):
        rpath = input
        rpath = rpath.replace('\n', '\\n')
        rpath = rpath.replace('\t', '\\t')
        # can't just replace all whitespaces as filenames and directories can
        # contain spaces
        # while iterating through each component, remove prefixes ('\n', '\t'),
        # then strip the remaining string
        return rpath

    def lengthLongestPath(self, input):
        """
        :type input: str
        :rtype: int
        """

        normalized_path = self.normalize_string(input)
        longest_pathname = 0
        pathname = [0] * 100
        elems = list(normalized_path.strip().split("\\"))
        t_count = 0

        for i, elem in enumerate(elems):
            component = elem.strip()
            if i == 0 and "." not in component:
                pathname[0] = len(component)
            elif component == "n":
                t_count = 0
            elif component == "t":
                t_count += 1
            elif "." in component:
                #
                if i == 0:
                    return len(component)
                ncomp = list(component[1:])
                for i in range(max(4, len(ncomp))):
                    if ncomp[i] == ' ':
                        ncomp[i] = ''
                    elif ncomp[i] != ' ':
                        break
                ncomp = "".join(ncomp)
                print(ncomp)
                t_count += 1
                pathname_length = 0
                for i in range(t_count):
                    pathname_length += pathname[i] + 1
                pathname_length += len(ncomp)
                longest_pathname = max(pathname_length, longest_pathname)
            else:
                # in this case, it is a directory with an 'n' or 't' prefixed
                ncomp = component[1:].strip()
                t_count += 1
                pathname[t_count] = len(ncomp)

        return longest_pathname

# solution
class Solution(object):

    def lengthLongestPath(self, input):
        maxlen = 0
        # using dictionary since unknown length of paths
        pathlen = {0: 0}
        # split into components (separated by new lines)
        for line in input.splitlines():
            # strip out the tabs on the left
            name = line.lstrip('\t')
            # find the 'depth' of the dir or file by determining # of tabs
            # stripped
            depth = len(line) - len(name)
            # if it's a file then
            if '.' in name:
                # update max length with current depth path length and
                # length of the filename
                maxlen = max(maxlen, pathlen[depth] + len(name))
            else:
                # otherwise update the path length for the subsequent depth
                # with the name of
                # the current path and one extra for a slash character
                pathlen[depth + 1] = pathlen[depth] + len(name) + 1
        return maxlen


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


'''
461. Hamming Distance
'''

class Solution(object):

    def binary(self, x):
        '''
        :type x: int
        :rtype: reversed list of binary digits
        '''
        b = []
        xb = x
        while xb > 0:
            b.append(xb % 2)
            xb = xb // 2
        return b

    def hammingDistance(self, x, y):
        """
        :type x: int
        :type y: int
        :rtype: int
        """
        '''
        xb = self.binary(x)
        yb = self.binary(y)
        ham_distance = 0
        minlen = min(len(xb), len(yb))
        maxlen = max(len(xb), len(yb))
        for i in range(minlen):
            if xb[i] != yb[i]:
                ham_distance += 1
        for i in range(minlen, maxlen):
            binary = None
            if x > y:
                binary = xb
            else:
                binary = yb
            if binary[i] == 1:
                ham_distance += 1
        return ham_distance
        '''

        '''
        one line solution
        return bin(x^y).count('1')
        '''

        # terse solution
        ans = 0
        while x or y:
            ans += (x % 2) ^ (y % 2)
            x /= 2
            y /= 2
        return ans


'''
344. Reverse String
'''

class Solution(object):
    def reverseString(self, s):
        """
        :type s: str
        :rtype: str
        """
        # string = list(s)
        # string.reverse()
        # return "".join(string)

        return s[::-1]


'''
448. Find All Numbers Disappeared in an Array
'''

class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """

        '''
        num_map = {}
        missing_nums = []
        for x in nums:
            num_map[x] = 1
        for i in range(1, len(nums) + 1):
            if i not in num_map:
                missing_nums.append(i)
        return missing_nums
        '''

        # For each number i in nums,
        # we mark the number that i points as negative.
        # Then we filter the list, get all the indexes
        # who points to a positive number
        for i in range(len(nums)):
            index = abs(nums[i]) - 1
            nums[index] = - abs(nums[index])

        return [i + 1 for i in range(len(nums)) if nums[i] > 0]


'''
463. Island Perimeter
'''

class Solution(object):
    def islandPerimeter(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """


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


'''
236. Lowest Common Ancestor of a Binary Tree
'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

from collections import deque

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """

        memo = {}
        return self.lcaHelper(root, p, q, memo)

    def lcaHelper(self, root, p, q, memo):

        queue = deque()
        queue.append(root)
        while len(queue) > 0:
            node = queue.popleft()

            lc, rc = 0, 0
            if node == None:
                continue
            if root.left != None:
                lc += self.lcaCount(root.left, p, q, memo)
            if root.right != None:
                rc += self.lcaCount(root.right, p, q, memo)
            if lc == 1 and rc == 1:
                return root
            if lc == 1 or rc == 1:
                return root
            elif lc == 2:
                queue.append(root.left)
            else:
                queue.append(root.right)

    def lcaCount(self, root, p, q, memo):

        if root == None:
            return 0

        if root.val in memo:
            return memo[root.val]

        count = self.lcaCount(root.left, p, q, memo) + self.lcaCount(
            root.right, p, q, memo)
        if root.val == p.val or root.val == q.val:
            count += 1
        memo[root.val] = count
        return memo[root.val]


'''
237. Delete Node in a Linked List
'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next = node.next.next

'''
141. Linked List Cycle
'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if not head:
            return False
        n1, n2 = head, head.next
        while n1 and n2:
            if n1.val == n2.val:
                return True
            n1 = n1.next
            if n2.next:
                n2 = n2.next.next
            else:
                return False
        return False

'''
191. Number of 1 Bits
'''

class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        count = 0
        for i in range(32):
            if ((1 << i) & n) != 0:
                count += 1
        return count

'''
346. Moving Average from Data Stream
'''

class MovingAverage(object):

    def __init__(self, size):
        """
        Initialize your data structure here.
        :type size: int
        """
        self._circularArray = [0 for i in range(size)]
        self._counter = 0
        self._sum = 0
        self._currIndex = 0

    def next(self, val):
        """
        :type val: int
        :rtype: float
        """
        self._counter += 1
        self._sum -= self._circularArray[self._currIndex]
        self._sum += val
        self._circularArray[self._currIndex] = val
        self._currIndex = (self._currIndex + 1) % len(self._circularArray)
        return float(self._sum) / min(len(self._circularArray), self._counter)

# Your MovingAverage object will be instantiated and called as such:
# obj = MovingAverage(size)
# param_1 = obj.next(val)

'''
155. Min Stack
'''

class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.items = []
        self.mins = []

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        self.items.append(x)
        if len(self.mins) == 0 or x <= self.mins[-1]:
            self.mins.append(x)

    def pop(self):
        """
        :rtype: void
        """
        poppedItem = self.items.pop()
        if len(self.mins) > 0 and poppedItem == self.mins[-1]:
            self.mins.pop()

    def top(self):
        """
        :rtype: int
        """
        return self.items[-1] if len(self.items) > 0 else 0

    def getMin(self):
        """
        :rtype: int
        """
        return self.mins[-1] if len(self.mins) > 0 else 0

# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()

'''
20. Valid Parentheses
'''

class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        brackets = {")":"(","}":"{","]":"["}
        for char in s:
            if char in brackets.values():
                stack.append(char)
            elif char in brackets.keys():''
                if len(stack) > 0 and stack[-1] == brackets[char]:
                    stack.pop()
                else:
                    return False
        return len(stack) == 0

'''
146. LRU Cache
'''

from collections import deque

class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self.queue = deque()
        self.cache = {}
        self.lives = {}
        self.count = 0

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key not in self.cache:
            return -1
        else:
            self.lives[key] += 1
            self.queue.append(key)
            return self.cache[key]

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        if key not in self.cache:
            self.count += 1
            if self.count > self.capacity:
                self.evict()
            self.lives[key] = 1
        else:
            self.lives[key] += 1
        self.queue.append(key)
        self.cache[key] = value

    def evict(self):
        while self.count > self.capacity:
            key = self.queue.popleft()
            self.lives[key] -= 1
            if self.lives[key] == 0:
                self.count -= 1
                del self.cache[key]

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

'''
326. Power of Three
'''

from math import *

class Solution(object):
    def isPowerOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        """
        return (log10(n)/log10(3)).is_integer() if n > 0 else False

'''
125. Valid Palindrome
'''

class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        s = [c.lower() for c in s if c.isalnum()]
        return all([s[i] == s[~i] for i in range(len(s) // 2)])

'''
21. Merge Two Sorted Lists
'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        '''
        # recursive
        if not l1 or not l2:
            return l1 or l2
        m = ListNode(0) # dummy head
        if l1.val < l2.val:
            m.next = l1
            m.next.next = self.mergeTwoLists(l1.next, l2)
        else:
            m.next = l2
            m.next.next = self.mergeTwoLists(l1, l2.next)
        return m.next
        '''
        # iterative
        m = ListNode(0)
        tail = m
        while l1 and l2:
            if l1.val < l2.val:
                tail.next, l1 = l1, l1.next
            else:
                tail.next, l2 = l2, l2.next
            tail = tail.next
        tail.next = l1 or l2
        return m.next


'''
110. Balanced Binary Tree
'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        height = self.isBalancedHelper(root, 0)
        return height >= 0

    def isBalancedHelper(self, root, height):
        '''
        :type root: TreeNode, height: int
        :rtype height: int
        '''
        if not root:
            return height
        lh = self.isBalancedHelper(root.left, height+1)
        rh = self.isBalancedHelper(root.right, height+1)
        if lh == -1 or rh == -1 or abs(lh-rh) > 1:
            return -1
        return max(lh, rh)

'''
121. Best Time to Buy and Sell Stock
'''

class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if len(prices) <= 1:
            return 0
        max_profit = float("-inf")
        min_seen = prices[0]
        for i, price in enumerate(prices[1:]):
            max_profit = max(max_profit, price - min_seen)
            min_seen = min(min_seen, price)
        return max(max_profit, 0)

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

'''
628. Maximum Product of Three Numbers
'''

import heapq
from functools import reduce
from operator import mul

class Solution(object):
    def maximumProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        [1, 2, 3] -> trivial, return all 3 numbers multiplied
        [7, 6, 2, 4]
        [0, 6, 2, 4]
        things to think about ... negative numbers and zeroes
        brute force is n * n * n O(n^3)
        max product will be
        three largest positive numbers
        two largest negative and largest positive
            largest negative and two positive leads to negative #
            three largest negative leads to negative
        maintain min and max heaps
        pop off smallest two and largest 3, find max between the two
        '''
        max_3, min_2 = nums[:3], list(map(lambda x: -x, nums[:2]))
        heapq.heapify(max_3)
        heapq.heapify(min_2)
        heapq.heappushpop(min_2, -nums[2])
        for i, x in enumerate(nums[3:]):
            heapq.heappushpop(max_3, x)
            heapq.heappushpop(min_2, -x)
        # print(max_3, min_2)
        pos_3 = functools.reduce(lambda t, x: t * x, max_3)
        neg_2 = -min_2[0] * -min_2[1]
        neg_2 *= max(max_3)
        # print(pos_3, neg_2)
        return max(pos_3, neg_2)

        '''
        concise solution
        '''

        max_3, min_2 = heapq.nlargest(3, nums), heapq.nsmallest(2, nums)
        return max(reduce(mul, max_3), reduce(mul, min_2 + [max_3[0]]))

'''
56. Merge Intervals
'''

# Definition for an interval.
# class Interval(object):
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        def overlap(t1, t2):
            '''
            :type t1: Interval, t2: Interval
            :rtype: Interval
            '''
            if t2.start < t1.start:
                t1, t2 = t2, t1
            if t2.start > t1.end:
                return None
            else:
                return Interval(t1.start, max(t1.end, t2.end))

        if len(intervals) < 2:
            return intervals
        intervals.sort(key=lambda x: (x.start, x.end))
        oi = 0
        cur = intervals[0]
        for interval in intervals[1:]:
            new_interval = overlap(cur, interval)
            if new_interval:
                cur = new_interval
            else:
                intervals[oi] = cur
                oi += 1
                cur = interval
        intervals[oi] = cur
        oi += 1
        return intervals[:oi]

        '''
        concise soln
        '''

        out = []
        intervals.sort(key=lambda x: x.start)
        for interval in intervals:
            if out and interval.start <= out[-1].end:
                out[-1].end = max(out[-1].end, interval.end)
            else:
                out.append(interval)
        return out

'''
518. Coin Change 2
'''

class Solution(object):
    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        combinations = [0] * (amount + 1)
        combinations[0] = 1

        for coin in coins:
            for x in range(coin, amount+1):
                remainder = x - coin
                combinations[x] += combinations[remainder]
        return combinations[amount]

'''
223. Rectangle Area
'''

class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def area(self):
        return self.w * self.h

class Line:
    def __init__(self, start, length):
        self.start = start
        self.length = length

class Solution(object):

    def computeArea(self, A, B, C, D, E, F, G, H):
        """
        :type A: int
        :type B: int
        :type C: int
        :type D: int
        :type E: int
        :type F: int
        :type G: int
        :type H: int
        :rtype: int
        """
        rect1, rect2 = Rect(A, B, C-A, D-B), Rect(E, F, G-E, H-F)
        area = rect1.area() + rect2.area()
        intersecting_rect = self.computeIntersectingRectangle(rect1, rect2)
        if intersecting_rect:
            area -= intersecting_rect.area()
        return area

    def computeIntersectingRectangle(self, rect1, rect2):
        '''
        :type rect1: Rect
        :type rect2: Rect
        :rtype: Rect
        '''
        hline1, hline2 = Line(rect1.x, rect1.w), Line(rect2.x, rect2.w)
        hline = self.intersectingLineSegment(hline1, hline2)
        if not hline:
            return
        vline1, vline2 = Line(rect1.y, rect1.h), Line(rect2.y, rect2.h)
        vline = self.intersectingLineSegment(vline1, vline2)
        if not vline:
            return
        return Rect(hline.start, vline.start, hline.length, vline.length)

    def intersectingLineSegment(self, line1, line2):
        if line2.start < line1.start:
            line1, line2 = line2, line1
        if line2.start > line1.start+line1.length:
            return
        return Line(line2.start,
                    min(line2.length,
                        line1.start+line1.length - line2.start))

'''
98. Validate Binary Search Tree
'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.isValidBSTHelper(root, float('-inf'), float('inf'))

    def isValidBSTHelper(self, root, minval, maxval):
        if not root:
            return True
        if root.val <= minval or root.val >= maxval:
            return False
        return (self.isValidBSTHelper(root.left, minval, root.val)
                and self.isValidBSTHelper(root.right, root.val, maxval))

'''
1. Two Sum
'''

class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        complements = {}
        for i, x in enumerate(nums):
            complement = target - x
            if complement in complements:
                return [complements[complement], i]
            else:
                complements[x] = i


'''
384. Shuffle an Array
'''

import random

class Solution(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.nums = nums

    def reset(self):
        """
        Resets the array to its original configuration and return it.
        :rtype: List[int]
        """
        return self.nums

    def shuffle(self):
        """
        Returns a random shuffling of the array.
        :rtype: List[int]
        """
        shuffled = self.nums[:]
        result_index = 0
        while result_index < len(self.nums) - 1:
            rand_index = random.randrange(result_index, len(self.nums))
            shuffled[result_index], shuffled[rand_index] = \
                shuffled[rand_index], shuffled[result_index]
            result_index += 1
        return shuffled

'''
155. Min Stack
'''

class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.items = []
        self.minstack = []

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        self.items.append(x)
        if not self.minstack.peek() or x <= self.minstack.peek():
            self.minstack.append(x)

    def pop(self):
        """
        :rtype: void
        """
        if not self.isEmpty():
            x = self.items.pop()
            if x == self.minstack[-1]:
                self.minstack.pop()
            return x

    def top(self):
        """
        :rtype: int
        """
        if not self.isEmpty():
            return self.items[-1]


    def getMin(self):
        """
        :rtype: int
        """
        if not self.isEmpty():
            return self.minstack[-1]

    def isEmpty(self):
        return not self.items

# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()

'''
104. Maximum Depth of Binary Tree
'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        def maxDepthHelper(root, depth):
            if not root:
                return depth
            return max(maxDepthHelper(root.left, depth+1),
                       maxDepthHelper(root.right, depth+1))
        return maxDepthHelper(root, 0)

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

'''
100. Same Tree
'''

def is_equal(t1, t2):
    stack1, stack2 = [t1], [t2]
    while stack1 and stack2:
        n1, n2 = stack1.pop(), stack2.pop()
        if (n1 and not n2) or (not n1 and n2):
            return False
        if n1 and n2 and n1.val != n2.val:
            return False
        if n1 and n2:
            stack1.extend([n1.left, n1.right])
            stack2.extend([n2.left, n2.right])
    return stack1 == stack2

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if p and q:
            return p.val == q.val and self.isSameTree(p.left, q.left) and \
                self.isSameTree(p.right, q.right))
        return p is q

'''
292. Nim Game
'''

class Solution(object):
    def canWinNim(self, n):
        """
        :type n: int
        :rtype: bool
        """
        return not n % 4 == 0

'''
136. Single Number
'''

import functools

class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return functools.reduce(lambda t, x: t ^ x, nums)

'''
101. Symmetric Tree
'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def traversal(t1, t2):
            if bool(t1) != bool(t2) or ((t1 and t2) and (t1.val != t2.val)):
                return False
            if not t1 and not t2:
                return True
            if not traversal(t1.left, t2.right) or \
                not traversal(t1.right, t2.left):
                    return False
            return True
        if root:
            return traversal(root.left, root.right)
        return True

        '''
        EPI solution
        '''

        def check_symmetric(t1, t2):
            if not t1 and not t2:
                return True
            elif t1 and t2:
                return t1.val == t2.val and \
                    check_symmetric(t1.left, t2.right) and \
                    check_symmetric(t1.right, t2.left)
            return False
        return not root or check_symmetric(root.left, root.right)

'''
237. Delete Node in a Linked List
'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next = node.next.next

'''
9. Palindrome Number
'''

class Solution(object):
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x < 0 or (x != 0 and x % 10 == 0):
            return False
        rev = 0
        while x > rev:
            rev, x = rev * 10 + x % 10, x // 10
        return x == rev or x == rev // 10

'''
236. Lowest Common Ancestor of a Binary Tree
'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

import collections

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        Status = collections.namedtuple("Status", ('num_nodes', 'ancestor'))

        def lca_helper(tree, n1, n2):
            if not tree:
                return Status(0, None)
            left = lca_helper(tree.left, n1, n2)
            if left.num_nodes == 2:
                return left
            right = lca_helper(tree.right, n1, n2)
            if right.num_nodes == 2:
                return right
            num_nodes = left.num_nodes + right.num_nodes + \
                int(tree is n1) + int(tree is n2)
            return Status(num_nodes, tree if num_nodes == 2 else None)

        return lca_helper(root, p, q).ancestor

'''
50. Pow(x, n)
'''

class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        result, power = 1.0, n
        if power < 0:
            power, x = -power, 1.0 / x
        while power:
            if power & 1:
                result *= x
            x *= x
            power >>= 1
        return result

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

'''
102. Binary Tree Level Order Traversal
'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

import collections

class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        output = []
        if not root:
            return output
        Item = collections.namedtuple('Item', ('node', 'depth'))
        q = collections.deque()
        q.append(Item(root, 0))
        while q:
            x = q.popleft()
            node = x.node
            if node:
                depth = x.depth
                if depth not in range(len(output)):
                    output.append([])
                output[depth].append(node.val)
                q.append(Item(node.left, depth + 1))
                q.append(Item(node.right, depth + 1))
        return output

'''
146. LRU Cache
'''

import collections

class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self.cache = collections.OrderedDict()

    def get(self, key):
        """
        Get the value (will always be positive) of the key if the key
        exists in the cache, otherwise return -1.
        :type key: int
        :rtype: int
        """
        if key not in self.cache:
            return -1
        val = self.cache.pop(key)
        self.cache[key] = val
        return val

    def put(self, key, value):
        """
         Set or insert the value if the key is not already present.
         When the cache reached its capacity, it should invalidate the
         least recently used item before inserting a new item.
        :type key: int
        :type value: int
        :rtype: void
        """
        if key in self.cache:
            self.cache.pop(key)
        elif self.capcity <= len(self.cache):
            self.cache.popitem(last=False)
        self.cache[key] = value

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

'''
Implementation without OrderedDict
'''

class Node:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.next = None
        self.prev = None

class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self.head = Node(0,0)
        self.tail = Node(0,0)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.cache = {}

    def get(self, key):
        """
        Get the value (will always be positive) of the key if the key
        exists in the cache, otherwise return -1.
        :type key: int
        :rtype: int
        """
        if key in self.cache:
            n = self.cache[key]
            self.remove(n)
            self.add(n)
            return n.val
        return -1

    def put(self, key, value):
        """
         Set or insert the value if the key is not already present.
         When the cache reached its capacity, it should invalidate the
         least recently used item before inserting a new item.
        :type key: int
        :type value: int
        :rtype: void
        """
        if key in self.cache:
            self.remove(self.cache[key])
        n = Node(key, value)
        self.add(n)
        self.cache[key] = n
        if len(self.cache) > self.capacity:
            n = self.head.next
            self.remove(n)
            del self.cache[n.key]

    def remove(self, node):
        p = node.prev
        n = node.next
        p.next = n
        n.prev = p

    def add(self, node):
        p = self.tail.prev
        p.next = node
        node.prev = p
        node.next = self.tail
        self.tail.prev = node

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

'''
344. Reverse String
'''

class Solution(object):
    def reverseString(self, s):
        """
        :type s: str
        :rtype: str
        """
        A = list(s)
        l, r = 0, len(s)-1
        while l < r:
            A[l], A[r] = A[r], A[l]
            l += 1
            r -= 1
        return "".join(A)

'''
151. Reverse Words in a String
'''

class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        s = " ".join(s.split())
        def reverse(s, l, r):
            while l < r:
                s[l], s[r] = s[r], s[l]
                l += 1
                r -= 1

        s = list(s)
        reverse(s, 0, len(s)-1)
        word_start_index = 0
        for i in range(len(s)+1):
            if i == len(s) or s[i] == ' ':
                reverse(s, word_start_index, i-1)
                word_start_index = i + 1

        return "".join(s)

'''
2. Add Two Numbers
'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

'''
0 0 1
2
-> 2 0 1

1
2
-> 3

5 5
6 6
-> 1 2 1

1
None
-> 1
'''

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        def appendDigit(n, x):
            newNode = ListNode(x)
            n.next = newNode
            return newNode

        dummyHead = ListNode(0)
        curr = dummyHead
        carryOver = 0
        while l1 or l2:
            tempSum = carryOver
            tempSum += l1.val if l1 else 0
            tempSum += l2.val if l2 else 0
            carryOver, digit = divmod(tempSum, 10)
            curr = appendDigit(curr, digit)
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
        if carryOver:
            curr = appendDigit(curr, carryOver)

        return dummyHead.next

'''
280. Wiggle Sort
'''

class Solution(object):
    def wiggleSort(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """

        def swap(A, i, j):
            A[i], A[j] = A[j], A[i]

        shouldBeLessThan = True
        for i in range(len(nums)-1):
            if shouldBeLessThan:
                if nums[i] > nums[i+1]:
                    swap(nums, i, i+1)
                shouldBeLessThan = False
            else:
                if nums[i] < nums[i+1]:
                    swap(nums, i, i+1)
                shouldBeLessThan = True

'''
461. Hamming Distance
'''

class Solution(object):
    def hammingDistance(self, x, y):
        """
        :type x: int
        :type y: int
        :rtype: int
        """
        xor = x ^ y
        diffBits = 0
        while xor:
            xor = xor & (xor - 1)
            diffBits += 1
        return diffBits

'''
371. Sum of Two Integers
'''

class Solution(object):
    def getSum(self, a, b):
        """
        :type a: int
        :type b: int
        :rtype: int
        """
        # 32 bits integer max
        MAX = 0x7FFFFFFF
        # 32 bits interger min
        MIN = 0x80000000
        # mask to get last 32 bits
        mask = 0xFFFFFFFF
        carryover = b
        while carryover != 0:
            digits = (a ^ b)
            carryover = (a & b) << 1
            a = digits & mask
            b = carryover & mask
        return a if a <= MAX else ~(a ^ mask)

'''
3. Longest Substring Without Repeating Characters
'''

class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        start = curmax = 0
        seen = {}
        for i, c in enumerate(s):
            if c in seen and start <= seen[c]:
                start = seen[c] + 1
            else:
                curmax = max(curmax, i - start + 1)
            seen[c] = i
        return curmax


'''
14. Longest Common Prefix
'''

class TrieNode:
    def __init__(self, x):
        self.val = x
        self.children = {}
        self.count = 0

    def childNode(self):
        if len(self.children) == 1
            return self.children[list(self.children.keys())[0]]
        else:
            return None

class Trie:
    def __init__(self):
        self.root = TrieNode(0)

    def insert(self, s):
        cur = self.root
        for i, c in enumerate(s):
            if c not in cur.children:
                cur.children[c] = TrieNode(c)
            cur = cur.children[c]
            cur.count += 1

    def lcp(self, root, num_strs):
        if not root or root.count != num_strs:
            return ""
        output = [root.val]
        if len(root.children) == 1:
            output.extend(self.lcp(root.childNode(), num_strs))
        return output

class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if len(strs) == 1:
            return strs[0]
        t = Trie()
        for s in strs:
            t.insert(s)
        output = t.lcp(t.root.childNode(), len(strs))
        return "".join(output) if output else ""

'''
simplified vertical scanning solution
'''

class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if not strs:
            return ""
        if len(strs) == 1:
            return strs[0]
        base = strs[0]
        for i, c in enumerate(base):
                for s in strs:
                    if i == len(s) or s[i] != c:
                        return base[:i]
        return base

'''
237. Delete Node in a Linked List
'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val, node.next = node.next.val, node.next.next

'''
11. Container With Most Water
'''

class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        i, j = 0, len(height)-1
        maxArea = 0
        while i < j:
            area = min(height[i], height[j]) * (j - i)
            maxArea = max(maxArea, area)
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
        return maxArea

'''
104. Maximum Depth of Binary Tree
'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        def maxDepthHelper(root, depth):
            if not root:
                return depth
            return max(maxDepthHelper(root.left, depth+1),
                       maxDepthHelper(root.right, depth+1))
        return maxDepthHelper(root, 0)

'''
7. Reverse Integer
'''

class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        max32 = 0x7fffffff
        rev = 0
        isNeg = x < 0
        x = abs(x)
        while x != 0:
            rev, x = rev * 10 + x % 10, x // 10
        if rev > max32:
            return 0
        return rev * -1 if isNeg else rev

'''
111. Minimum Depth of Binary Tree
'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

import collections

class Solution(object):
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        q = collections.deque()
        q.append((root, 1))
        while q:
            n, d = q.popleft()
            if not n.left and not n.right:
                return d
            if n.left:
                q.append((n.left, d+1))
            if n.right:
                q.append((n.right, d+1))

'''
108. Convert Sorted Array to Binary Search Tree
'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if not nums:
            return None
        if len(nums) == 1:
            return TreeNode(nums[0])
        mid = len(nums) // 2
        root = TreeNode(nums[mid])
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid+1:])
        return root

'''
366. Find Leaves of Binary Tree
'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def findLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return None
        output = []
        q = collections.deque()
        q.append((root, 0))
        while q:
            n, d = q.popleft()
            if not n:
                continue
            if d >= len(output):
                output.append([])
            output[d].append(n.val)
            q.append((n.left, d+1))
            q.append((n.right, d+1))
        return output.reverse()

'''
53. Maximum Subarray
'''

class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return None
        maxSum, curSum = float('-inf'), 0
        for x in nums:
            curSum += x
            maxSum = max(maxSum, curSum)
            if curSum < 0:
                curSum = 0
        return maxSum

'''
72. Edit Distance
'''

class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        def score(i, j):
            return 0 if word1[i-1] == word2[j-1] else 1
        n, m = len(word1)+1, len(word2)+1
        dp = [ [i] + [j if i == 0 else 0 for j in range(1, m)] \
            for i in range(n)]
        for i in range(1, n):
            for j in range(1, m):
                dp[i][j] = min(
                    dp[i-1][j-1] + score(i,j),
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1
                )
        return dp[n-1][m-1]

'''
70. Climbing Stairs
'''

class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 1:
            return 1
        if n == 2:
            return 2
        dp = [1 for _ in range(n+1)]
        dp[2] = 2
        for i in range(2, n+1):
            dp[i] = dp[i-2] + dp[i-1]
        return dp[n]

'''
198. House Robber
'''

class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        if len(nums) <= 1:
            return nums[0]
        if len(nums) <= 2:
            return max(nums[0], nums[1])
        # have at least 3 elements in the array at this point
        dp = [x for x in nums]
        for i, x in enumerate(nums[2:], 2):
            # skip current and take previous
            # if taking current, you're
            dp[i] = max(dp[i-1],
                        dp[i-2]+x,
                        dp[i-3]+x if i >= 3 else float('-inf'))
        return dp[len(nums)-1]

'''
300. Longest Increasing Subsequence
'''

class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        lengths = [0 for _ in nums]
        for i, x in enumerate(nums):
            lengths[i] = max(
                [lengths[j] if nums[j] < x else 0 for j in range(i)] + [0]
            ) + 1
        return max(lengths)

'''
338. Counting Bits
'''

class Solution(object):
    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """
        cache = {}
        def countBitsHelper(num):
            if num == 0:
                return 0
            if num in cache:
                return cache[num]
            cache[num] = countBitsHelper((num & (num - 1))) + 1
            return cache[num]

        return [countBitsHelper(i) for i in range(num+1)]


'''
139. Word Break
'''

class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        cache = {}

        def isWord(s):
            return s in wordDict

        def isPartionable(i):
            if i == len(s):
                return True
            if i in cache:
                return cache[i]
            for j in range(i, len(s)):
                candidate_word = s[i:j+1]
                if isWord(candidate_word) and isPartionable(j+1):
                    cache[i] = True
                    return cache[i]
            cache[i] = False
            return cache[i]

        return isPartionable(0)

'''
152. Maximum Product Subarray
'''

class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        curMax = curMin = maxSeen = nums[0]
        for x in nums[1:]:
            temp = curMax
            curMax = max(temp * x, curMin * x, x)
            curMin = min(temp * x, curMin * x, x)
            maxSeen = max(maxSeen, curMax)
        return maxSeen

'''
47. Permutations II
'''

class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if len(nums) == 1:
            return [nums]
        output = []
        seen = {}
        for i in range(len(nums)):
            if nums[i] not in seen:
                seen[nums[i]] = True
                nums[0], nums[i] = nums[i], nums[0]
                for p in self.permuteUnique(nums[1:]):
                    output.append([nums[0]] + p)
                nums[0], nums[i] = nums[i], nums[0]
        return output

'''
226. Invert Binary Tree
'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """

        if not root:
            return
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root

'''
217. Contains Duplicate
'''

from collections import defaultdict
class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        seen = defaultdict(int)
        for x in nums:
            seen[x] += 1
            if seen[x] > 1:
                return True
        return False

'''
328. Odd Even Linked List
'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        oddDummy = oddTail = ListNode(0)
        evenDummy = evenTail = ListNode(0)
        while head:
            oddTail.next = head
            oddTail = oddTail.next
            evenTail.next = head.next
            evenTail = evenTail.next
            head = head.next.next if head.next else None
        oddTail.next = evenDummy.next
        return oddDummy.next

'''
88. Merge Sorted Array
'''

class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        i = m + n - 1
        m -= 1
        n -= 1
        while i >= 0 and m >= 0 and n >= 0:
            if nums1[m] < nums2[n]:
                maxVal = nums2[n]
                n -= 1
            else:
                maxVal = nums1[m]
                m -= 1
            nums1[i] = maxVal
            i -= 1
        while m >= 0:
            nums1[i] = nums1[m]
            i -= 1
            m -= 1
        while n >= 0:
            nums1[i] = nums2[n]
            i -= 1
            n -= 1
