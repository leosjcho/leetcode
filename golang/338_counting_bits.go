/*
338. Counting Bits
*/

/*
Given a non negative integer number num. For every numbers i in the range
0 ≤ i ≤ num calculate the number of 1's in their binary representation and
return them as an array.

Example:
For num = 5 you should return [0,1,1,2,1,2].

0000 0
0001 1
0010 1
0011 2
0100 1
0101 2
0110 2
0111 3
1000         1
1001 1 + 1 = 2
1010 1 + 1 = 2
1011 1 + 2 = 3
1100 1 + 1 = 2
1101 1 + 2 = 3
1110 1 + 2 = 3
1111 1 + 3 = 4
*/

func countBits(num int) []int {
	dp := make([]int, num+1)
	for i := 1; i <= num; i++ {
		dp[i] = dp[i>>1] + (i & 1)
	}
	return dp
}

