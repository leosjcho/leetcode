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

