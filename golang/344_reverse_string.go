/*
344. Reverse String
*/

/*
empty string:
i = 0, j = 0-1
i is not less than j. returns empty string
"a"
i = 0, j = 1-1 = 0
i is not less than j
returns "a"
"ab"
i = 0, j = 2-1 = 1
i is less than j.
swaps i and j
i = 1, j = 0
i is not less than j
returns "ba"
*/

func reverseString(s string) string {
	i, j := 0, len(s)-1
	r := []rune(s)
	for i < j {
		r[i], r[j] = r[j], r[i]
		i++
		j--
	}
	return string(r)
}

