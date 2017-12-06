/*
20. Valid Parentheses
*/

func isValid(s string) bool {
	stack := NewAStack()
	for _, c := range s {
		sc := string(c)
		if strings.Contains("({[", sc) {
			stack.Push(sc)
		} else {
			if stack.Peek() != matchingOpen(sc) {
				return false
			}
			stack.Pop()
		}
	}
	return stack.Size() == 0
}

func matchingOpen(s string) string {
	if s == ")" {
		return "("
	} else if s == "}" {
		return "{"
	} else {
		return "["
	}
}

