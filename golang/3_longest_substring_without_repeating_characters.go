/*
3. Longest Substring Without Repeating Characters
*/

func lengthOfLongestSubstring(s string) int {
	start, maxLength := 0, 0
	usedChar := map[byte]int{}
	for i := 0; i < len(s); i++ {
		_, seen := usedChar[s[i]]
		if seen && start <= usedChar[s[i]] {
			start = usedChar[s[i]] + 1
		} else {
			len := i - start + 1
			if len > maxLength {
				maxLength = len
			}
		}
		usedChar[s[i]] = i
	}
	return maxLength
}

