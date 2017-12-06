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

