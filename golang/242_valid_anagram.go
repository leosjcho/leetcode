/*
242. Valid Anagram
*/

func isAnagram(s string, t string) bool {
	chars := map[rune]int{}
	for _, c := range s {
		chars[c]++
	}
	for _, c := range t {
		chars[c]--
	}
	for _, count := range chars {
		if count != 0 {
			return false
		}
	}
	return true
}

