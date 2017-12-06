'''
49. Group Anagrams
'''

from collections import defaultdict

class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        sorted_strs = ["".join(sorted(s)) for s in strs]
        buckets = defaultdict(list)
        for i, s in enumerate(sorted_strs):
            buckets[s] = buckets[s] + [i]
        anagrams = []
        for bucket in buckets:
            indices = buckets[bucket]
            group = []
            for i in indices:
                group.append(strs[i])
            anagrams.append(group)
        return anagrams

