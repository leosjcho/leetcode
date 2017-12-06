/*
8. String to Integer (atoi)
*/

/*
questions:
what happens if a string has a non-numeric character? what should we return?
if it precedes the integral region, return 0
if it follows the integral region, ignore
multiple zeros?
what if the string causes integer overflow? - return int max, int min
negative values? "-00001"
whitespace characters?
*/

/*
first accepted
*/
/*
func myAtoi(str string) int {
	str = strings.TrimLeftFunc(str, func(arg2 rune) bool {
		return unicode.IsSpace(arg2)
	})
	if len(str) == 0 {
		return 0
	}
	pos := true
	if str[0] == '-' {
		pos = false
	}
	if str[0] == '+' || str[0] == '-' {
		str = str[1:]
		if len(str) == 0 {
			return 0
		}
	}
	var rval int
	if hasPreIntegralRegion(str) {
		return 0
	}

	start, end := -1, len(str)-1

	for i, char := range str {
		isInt := isInt(char)
		if start == -1 && !isInt {
			return 0
		}
		if isInt && start == -1 {
			start = i
		}
		if !isInt {
			end = i - 1
			break
		}
	}

	mul := 1
	for i := end; i >= start; i-- {
		val := intFromChar(str[i]) * mul
		nrval := rval + val
		if nrval > 2147483647 {
			// integer overflow!
			if pos {
				return 2147483647
			}
			return -2147483648
		}
		rval = nrval
		mul *= 10
	}
	if !pos {
		rval *= -1
	}
	return rval
}

func intFromChar(c byte) int {
	return int(c - '0')
}

func isInt(c rune) bool {
	return (c >= '0' && c <= '9')
}

func hasPreIntegralRegion(str string) bool {
	for _, c := range str {
		if !isInt(c) {
			return true
		}
		break
	}
	return false
}
*/

/*
refactored accepted
*/

func myAtoi(str string) int {
	sign, base, i := 1, 0, 0
	for i < len(str) && str[i] == ' ' {
		i++
	}
	if i < len(str) && (str[i] == '-' || str[i] == '+') {
		if str[i] == '-' {
			sign = -1
		}
		i++
	}
	for i < len(str) && (str[i] >= '0' && str[i] <= '9') {
		if base > math.MaxInt32/10 || (base == math.MaxInt32/10 &&
			str[i]-'0' > 7) {
			if sign == 1 {
				return math.MaxInt32
			}
			return math.MinInt32
		}
		base = 10*base + int(str[i]-'0')
		i++
	}
	return base * sign
}

