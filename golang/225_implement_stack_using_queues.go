/*
225. Implement Stack using Queues
*/
type MyStack struct {
	q *Queue
}

/** Initialize your data structure here. */
func Constructor() MyStack {
	return MyStack{q: NewQueue()}
}

/** Push element x onto stack. */
func (this *MyStack) Push(x int) {
	this.q.Push(x)
	for i := 0; i < this.q.Size()-1; i++ {
		this.q.Push(this.q.Peek())
		this.q.Pop()
	}
}

/** Removes the element on top of the stack and returns that element. */
func (this *MyStack) Pop() int {
	return this.q.Pop()
}

/** Get the top element. */
func (this *MyStack) Top() int {
	return this.q.Peek()
}

/** Returns whether the stack is empty. */
func (this *MyStack) Empty() bool {
	return this.q.IsEmpty()
}

/**
 * Your MyStack object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Push(x);
 * param_2 := obj.Pop();
 * param_3 := obj.Top();
 * param_4 := obj.Empty();
 */

