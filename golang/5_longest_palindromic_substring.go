/*
5. Longest Palindromic Substring
*/

func longestPalindrome(s string) string {
	longest := 0
	ll, lr := 0, 0
	for i := 0; i < len(s); i++ {
		singleCount, sl, sr := palindromicSize(s, i, i)
		dblCount, dl, dr := palindromicSize(s, i, i+1)
		if singleCount > longest {
			longest = singleCount
			ll, lr = sl, sr
		}
		if dblCount > longest {
			longest = dblCount
			ll, lr = dl, dr
		}
	}
	return s[ll : lr+1]
}

func palindromicSize(s string, i, j int) (int, int, int) {
	if j >= len(s) || s[i] != s[j] {
		return 0, i, j
	}
	for i >= 0 && j < len(s) && s[i] == s[j] {
		i--
		j++
	}
	return j - i - 1, i + 1, j - 1
}

