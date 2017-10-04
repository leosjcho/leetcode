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
        if not self.minstack:
            self.minstack.append(x)
        elif x <= self.minstack[-1]:
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
