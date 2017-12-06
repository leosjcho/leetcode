/*
231. Power of Two
*/

func isPowerOfTwo(n int) bool {
	if n < 0 {
		return false
	}
	count := 0
	var i uint
	for i = 0; i < 32; i++ {
		if ((1 << i) & n) != 0 {
			count++
		}
	}
	return count == 1
}

