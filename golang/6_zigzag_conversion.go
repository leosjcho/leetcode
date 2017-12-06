/*
6. ZigZag Conversion
*/

func convert(s string, numRows int) string {
	if numRows < 2 {
		return s
	}
	rows := generateRows(s, numRows) // [][]byte
	zigZagged := []byte{}
	for i := range rows {
		zigZagged = append(zigZagged, rows[i]...)
	}
	return string(zigZagged)
}

func generateRows(s string, numRows int) [][]byte {
	rows := init2DByteSlice(numRows, 0)
	i, n := 0, len(s)
	for i < n {
		scan(rows, s, &i, true)
		scan(rows, s, &i, false)
	}
	return rows
}

func scan(rows [][]byte, s string, i *int, down bool) {
	for j := 0; j < len(rows)-1; j++ {
		var x int
		if down {
			x = j
		} else {
			x = len(rows) - 1 - j
		}
		if *i == len(s) {
			return
		}
		rows[x] = append(rows[x], s[*i])
		*i++
	}
}

func init2DByteSlice(rows, cols int) [][]byte {
	arr := make([][]byte, rows)
	for i := range arr {
		arr[i] = make([]byte, cols)
	}
	return arr
}

