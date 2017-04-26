package main

// 1. Max Profit

func getMaxProfit(prices []int) int {
	if len(prices) < 2 {
		return 0
	}
	lowestSeen, maxProfit := prices[0], 0
	for _, price := range prices[1:] {
		profit := price - lowestSeen
		maxProfit = max(maxProfit, profit)
		lowestSeen = min(lowestSeen, price)
	}
	return maxProfit
}

/*
Test Cases:
simple:
1 2 3
lowest seen = 1
max = 0
price = 2
profit = 2 - 1 = 1
max = 1
price = 3
profit = 3 - 1
profit = 2
return 2

3 2 1
lowest seen = 3
maxProfit = 0
profit = 2 - 3 = -1
lowest seen = 2
profit = 1 - 2 = -1
lowest seen = 1
return 0

2 3 1
profit = 3 - 2 = 1
max = 1
profit = 1 - 2 = -1
return 1

empty
return 0

1
return 0
*/

// 2. getProductsOfAllIntsExceptAtIndex

// [ 1, 2, 3 ]
// [ 2 * 3, 1 * 3, 1 * 2 ]

// special cases
// empty array? return empty array

// one value?
// [ 1 ]
// [ 0 ] ?

func getProductsOfAllIntsExceptAtIndex(nums []int) []int {
	results := make([]int, n)
	runningTotal := 1
	for i := range nums {
		results[i] = runningTotal
		runningTotal *= nums[i]
	}
	runningTotal = 1
	for i := len(nums) - 1; i >= 0; i-- {
		results[i] *= runningTotal
		runningTotal *= nums[i]
	}
	return results
}
