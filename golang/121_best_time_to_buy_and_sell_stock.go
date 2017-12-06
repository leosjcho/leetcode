/*
121. Best Time to Buy and Sell Stock
*/

func maxProfit(prices []int) int {
	maxProfit := 0
	minSeen := math.MaxInt32
	for _, p := range prices {
		maxProfit = max(maxProfit, p-minSeen)
		minSeen = min(minSeen, p)
	}
	return maxProfit
}

