package lru

import (
	"container/list"
	"log"
)

type LRUCache struct {
	data        map[int]*info
	accessOrder *list.List
	capacity    int
}

type info struct {
	v int
	e *list.Element
}

func Constructor(capacity int) LRUCache {
	return LRUCache{
		data:        map[int]*info{},
		accessOrder: list.New(),
		capacity:    capacity,
	}
}

// updates timestamp
// never evicts elements
func (this *LRUCache) Get(key int) int {
	i, ok := this.data[key]
	if !ok || i == nil || i.v == -1 {
		return -1
	}
	this.accessOrder.MoveToBack(i.e)
	return i.v
}

// update timestamp
// insert key
// evict if at capacity
func (this *LRUCache) Put(key int, value int) {
	i, ok := this.data[key]
	// is this a new item?
	if !ok || i == nil || i.v == -1 {
		e := this.accessOrder.PushBack(key)
		i = &info{v: value, e: e}
		this.data[key] = i
	} else {
		this.accessOrder.MoveToBack(i.e)
		i.v = value
	}

	if this.accessOrder.Len() > this.capacity {
		e := this.accessOrder.Front()
		key, ok := e.Value.(int)
		if !ok {
			log.Fatal("invalid list element")
		}
		this.data[key] = nil
		this.accessOrder.Remove(e)
	}
}

// faster implementation but more space usage

/*
type LRUCache struct {
	data     map[int]int
	lives    map[int]int
	purgeQ   []int // queue
	count    int
	capacity int
}

func Constructor(capacity int) LRUCache {
	return LRUCache{
		data:     map[int]int{},
		lives:    map[int]int{},
		purgeQ:   []int{},
		count:    0,
		capacity: capacity,
	}
}

// updates timestamp
// never evicts elements
func (this *LRUCache) Get(key int) int {
	v, ok := this.data[key]
	if !ok {
		return -1
	}
	if v != -1 {
	    this.lives[key]++
	    this.purgeQ = append(this.purgeQ, key)
	}
	return v
}

// update timestamp
// insert key
// evict if at capacity
func (this *LRUCache) Put(key int, value int) {
	v, ok := this.data[key]
	// is this a new item?
	if !ok || v == -1 {
		this.count++
	}

	this.lives[key]++
	this.purgeQ = append(this.purgeQ, key)
	this.data[key] = value

	for this.count > this.capacity {
		candidate := this.purgeQ[0]
		this.purgeQ = this.purgeQ[1:]
		this.lives[candidate]--
		if this.lives[candidate] == 0 {
			// evict!
			this.data[candidate] = -1
			this.count--

		}
	}
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * obj := Constructor(capacity);
 * param_1 := obj.Get(key);
 * obj.Put(key,value);
*/
