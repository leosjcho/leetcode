package golang

/*
Data Structures
*/

type Queue struct {
	items []int
}

func NewQueue() *Queue {
	return &Queue{items: []int{}}
}

func (q *Queue) Push(x int) {
	q.items = append(q.items, x)
}

func (q *Queue) Peek() int {
	if q.Size() == 0 {
		return -1
	}
	return q.items[0]
}

func (q *Queue) Pop() int {
	if q.Size() == 0 {
		return -1
	}
	x := q.items[0]
	q.items = q.items[1:]
	return x
}

func (q *Queue) Size() int {
	return len(q.items)
}

func (q *Queue) IsEmpty() bool {
	return len(q.items) == 0
}

type AStack struct {
	items []string
}

func NewAStack() *AStack {
	return &AStack{[]string{}}
}

func (a *AStack) Push(s string) {
	a.items = append(a.items, s)
}

func (a *AStack) Pop() string {
	if len(a.items) == 0 {
		return ""
	}
	val := a.items[len(a.items)-1]
	a.items = a.items[:len(a.items)-1]
	return val
}

func (a *AStack) Peek() string {
	if len(a.items) == 0 {
		return ""
	}
	return a.items[len(a.items)-1]
}

func (a *AStack) Size() int {
	return len(a.items)
}

type IStack struct {
	items []int
}

func NewIStack() *IStack {
	return &IStack{[]int{}}
}

func (s *IStack) Push(x int) {
	s.items = append(s.items, x)
}

func (s *IStack) Pop() int {
	if len(s.items) == 0 {
		return -1
	}
	val := s.items[len(s.items)-1]
	s.items = s.items[:len(s.items)-1]
	return val
}

func (s *IStack) Peek() int {
	if len(s.items) == 0 {
		return -1
	}
	return s.items[len(s.items)-1]
}

func (s *IStack) Size() int {
	return len(s.items)
}
