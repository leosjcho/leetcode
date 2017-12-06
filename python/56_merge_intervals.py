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

