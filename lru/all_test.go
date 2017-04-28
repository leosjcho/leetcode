package lru

import "testing"

func Test(t *testing.T) {
	obj := Constructor(2)
	obj.Put(1, 1)
	obj.Put(2, 2)
	obj.Put(3, 3)
	if obj.Get(1) != -1 {
		t.Error()
	}
	obj.Get(1)
}
