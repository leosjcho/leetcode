/*
7. Reverse Integer
*/

func reverse(x int) int {
	result := 0
	for x != 0 {
		tail := x % 10
		nResult := result*10 + tail
		if nResult > math.MaxInt32 || nResult < math.MinInt32 {
			return 0
		}
		result = nResult
		x /= 10
	}
	return result
}

