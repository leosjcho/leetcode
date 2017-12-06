/*
13. Roman to Integer
*/

func romanToInt(s string) int {
	values := map[byte]int{
		'I': 1,
		'V': 5,
		'X': 10,
		'L': 50,
		'C': 100,
		'D': 500,
		'M': 1000,
	}
	result := values[s[len(s)-1]]
	for i := len(s) - 2; i >= 0; i-- {
		v := values[s[i]]
		if v < values[s[i+1]] {
			result -= v
		} else {
			result += v
		}
	}
	return result
}

