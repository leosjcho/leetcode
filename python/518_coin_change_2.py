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

