/*
155. Min Stack
*/

type MinStack struct {
	items []int
	mins  []int
}

/** initialize your data structure here. */
// func Constructor() MinStack {
// 	return MinStack{
// 		items: []int{},
// 		mins:  []int{},
// 	}
// }

func (this *MinStack) Push(x int) {
	m := len(this.mins)
	if m == 0 {
		this.mins = append(this.mins, x)
	} else if x <= this.mins[m-1] {
		this.mins = append(this.mins, x)
	}
	this.items = append(this.items, x)
}

func (this *MinStack) Top() int {
	return this.items[len(this.items)-1]
}

func (this *MinStack) Pop() {
	n := len(this.items)
	m := len(this.mins)
	item := this.items[n-1]
	// pop from items stack
	this.items = this.items[:n-1]
	// if the item was a min value
	if this.mins[m-1] == item {
		// pop from mins stack
		this.mins = this.mins[:m-1]
	}
}

func (this *MinStack) GetMin() int {
	return this.mins[len(this.mins)-1]
}

/**
 * Your MinStack object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Push(x);
 * obj.Pop();
 * param_3 := obj.Top();
 * param_4 := obj.GetMin();
 */

