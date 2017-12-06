/*
266. Palindrome Permutation
*/

func canPermutePalindrome(s string) bool {
	seen := map[rune]bool{}
	for _, c := range s {
		_, ok := seen[c]
		if ok {
			delete(seen, c)
		} else {
			seen[c] = true
		}
	}
	if len(s)%2 == 0 {
		return len(seen) == 0
	}
	return len(seen) == 1
}

