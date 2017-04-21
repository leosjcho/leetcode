package main

import "strings"

/*
461
Hamming Distance
*/

func hammingDistance(x int, y int) int {
	count := 0
	var i uint
	xor := uint(x) ^ uint(y)
	for i = 0; i < 32; i++ {
		if ((1 << i) & xor) != 0 {
			count++
		}
	}
	return count
}

/*
557
Reverse Words in a String III
*/

func reverseWords(s string) string {
	rw := []string{}
	ws := strings.Split(s, " ")
	for _, w := range ws {
		rw = append(rw, reverseWord(w))
	}
	return strings.Join(rw, " ")
}

func reverseWord(s string) string {
	b := []byte(s)
	for i := 0; i < len(b)/2; i++ {
		b[i], b[len(b)-1-i] = b[len(b)-1-i], b[i]
	}
	return string(b)
}
