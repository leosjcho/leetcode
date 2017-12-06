/*
367. Valid Perfect Square
*/

func isPerfectSquare(num int) bool {
	lo, hi := 0, math.MaxUint32
	for lo <= hi {
		mid := (hi-lo)/2 + lo
		if num == mid*mid {
			return true
		}
		if num < mid*mid {
			hi = mid - 1
		} else {
			lo = mid + 1
		}
	}
	return false
}

