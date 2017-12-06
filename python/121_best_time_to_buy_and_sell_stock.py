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

