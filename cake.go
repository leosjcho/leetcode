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

/*
35. In-place Shuffle
*/

func inPlaceShuffle(items []int) {
	for i := range len(items) {
		j := getRandom(i, len(items)-1)
		if i != j {
			items[i], items[j] = items[j], items[i]
		}
	}
}

/*
36. Single Riffle Shuffle
*/

func isSingleRiffleRecursive(half1, half2, shuffledDeck []int) bool {
	if len(shuffledDeck) == 0 {
		return true
	}
	card := shuffledDeck[0]
	if len(half1) > 0 && len(half2) > 0 && card == half1[0] && card == half2[0] {
		return isSingleRiffle(half1[1:], half2, shuffledDeck[1:]) ||
			isSingleRiffle(half1, half2[1:], shuffledDeck[1:])
	}
	if len(half1) > 0 && card == half1[0] {
		return isSingleRiffle(half1[1:], half2, shuffledDeck[1:])
	}
	if len(half2) > 0 && card == half2[0] {
		return isSingleRiffle(half1, half2[1:], shuffledDeck[1:])
	}
	return false
}

func singleRiffleIterative(half1, half2, shuffledDeck []int) bool {
	h1, h2 := 0, 0
	for _, card := range shuffledDeck {
		if card == half1[h1] {
			h1++
		}
	}
}
