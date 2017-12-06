/*
232. Implement Queue using Stacks
*/

type MyQueue struct {
	input  *IStack
	output *IStack
}

/** Initialize your data structure here. */
// func Constructor() MyQueue {
// 	return MyQueue{NewIStack(), NewIStack()}
// }

/** Push element x to the back of queue. */
func (this *MyQueue) Push(x int) {
	this.input.Push(x)
}

/** Removes the element from in front of queue and returns that element. */
func (this *MyQueue) Pop() int {
	this.Peek()
	return this.output.Pop()
}

/** Get the front element. */
func (this *MyQueue) Peek() int {
	if this.output.Size() == 0 {
		for this.input.Size() > 0 {
			this.output.Push(this.input.Pop())
		}
	}
	return this.output.Peek()
}

/** Returns whether the queue is empty. */
func (this *MyQueue) Empty() bool {
	return this.input.Size() == 0 && this.output.Size() == 0
}

/**
 * Your MyQueue object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Push(x);
 * param_2 := obj.Pop();
 * param_3 := obj.Peek();
 * param_4 := obj.Empty();
 */

