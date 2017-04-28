package lru

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
