/*
14. Longest Common Prefix
*/

func longestCommonPrefix(strs []string) string {
	lcp := []byte{}
	var curr byte
	i := 0
Loop:
	for {
		if len(strs) == 0 || i == len(strs[0]) {
			break
		}
		curr = strs[0][i]
		for _, str := range strs {
			if i == len(str) || curr != str[i] {
				break Loop
			}
		}
		lcp = append(lcp, curr)
		i++
	}
	return string(lcp)
}
