/*
4. Median of Two Sorted Arrays
*/

// https://discuss.leetcode.com/topic/28602/concise-java-solution-based-on-
// binary-search

func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	m, n := len(nums1), len(nums2)
	l, r := (m+n+1)/2, (m+n+2)/2
	return (getKth(nums1, nums2, 0, 0, l) + getKth(nums1, nums2, 0, 0, r)) / 2
}

func getKth(nums1, nums2 []int, aStart, bStart, k int) float64 {
	if aStart > len(nums1)-1 {
		return float64(nums2[bStart+k-1])
	}
	if bStart > len(nums2)-1 {
		return float64(nums1[aStart+k-1])
	}
	if k == 1 {
		return float64(min(nums1[aStart], nums2[bStart]))
	}
	aMid, bMid := math.MaxInt32, math.MaxInt32
	if aStart+k/2-1 < len(nums1) {
		aMid = nums1[aStart+k/2-1]
	}
	if bStart+k/2-1 < len(nums2) {
		bMid = nums2[bStart+k/2-1]
	}
	if aMid < bMid {
		return getKth(nums1, nums2, aStart+k/2, bStart, k-k/2)
	} else {
		return getKth(nums1, nums2, aStart, bStart+k/2, k-k/2)
	}
}

