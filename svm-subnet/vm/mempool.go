package vm

import (
	"container/heap"
	"errors"
	"sync"
)

var (
	ErrMempoolFull     = errors.New("mempool is full")
	ErrDuplicateTx     = errors.New("duplicate transaction")
	ErrInvalidTx       = errors.New("invalid transaction")
)

// Mempool manages pending transactions
type Mempool struct {
	mu sync.RWMutex

	// Max number of transactions
	maxSize int

	// Pending transactions by priority
	pending *txHeap

	// Transaction signatures for dedup
	seen map[string]struct{}
}

// NewMempool creates a new mempool
func NewMempool(maxSize int) *Mempool {
	h := &txHeap{}
	heap.Init(h)

	return &Mempool{
		maxSize: maxSize,
		pending: h,
		seen:    make(map[string]struct{}),
	}
}

// Add adds a transaction to the mempool
func (m *Mempool) Add(tx *Transaction) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Validate transaction
	if err := tx.Verify(); err != nil {
		return ErrInvalidTx
	}

	// Check for duplicate
	sig := string(tx.Signature())
	if _, exists := m.seen[sig]; exists {
		return ErrDuplicateTx
	}

	// Check capacity
	if m.pending.Len() >= m.maxSize {
		return ErrMempoolFull
	}

	// Add to pending
	heap.Push(m.pending, &txItem{
		tx:       tx,
		priority: m.calculatePriority(tx),
	})
	m.seen[sig] = struct{}{}

	return nil
}

// Pop removes and returns the highest priority transaction
func (m *Mempool) Pop() *Transaction {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.pending.Len() == 0 {
		return nil
	}

	item := heap.Pop(m.pending).(*txItem)
	delete(m.seen, string(item.tx.Signature()))
	return item.tx
}

// PopAll removes and returns up to n transactions
func (m *Mempool) PopAll(n int) []*Transaction {
	m.mu.Lock()
	defer m.mu.Unlock()

	var result []*Transaction
	for i := 0; i < n && m.pending.Len() > 0; i++ {
		item := heap.Pop(m.pending).(*txItem)
		delete(m.seen, string(item.tx.Signature()))
		result = append(result, item.tx)
	}

	return result
}

// Len returns the number of pending transactions
func (m *Mempool) Len() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.pending.Len()
}

// Has checks if a transaction is in the mempool
func (m *Mempool) Has(signature []byte) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	_, exists := m.seen[string(signature)]
	return exists
}

// Remove removes a transaction from the mempool
func (m *Mempool) Remove(signature []byte) {
	m.mu.Lock()
	defer m.mu.Unlock()

	sig := string(signature)
	if _, exists := m.seen[sig]; !exists {
		return
	}

	delete(m.seen, sig)

	// Find and remove from heap (O(n) but mempool is typically small)
	for i := 0; i < m.pending.Len(); i++ {
		if string((*m.pending)[i].tx.Signature()) == sig {
			heap.Remove(m.pending, i)
			break
		}
	}
}

// Clear removes all transactions from the mempool
func (m *Mempool) Clear() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.pending = &txHeap{}
	heap.Init(m.pending)
	m.seen = make(map[string]struct{})
}

// calculatePriority computes transaction priority
// Higher priority = processed first
func (m *Mempool) calculatePriority(tx *Transaction) uint64 {
	// TODO: Calculate based on fee, compute units requested, etc.
	// For now, use a simple FIFO approach (constant priority)
	return 0
}

// txItem wraps a transaction with its priority for the heap
type txItem struct {
	tx       *Transaction
	priority uint64
	index    int
}

// txHeap implements heap.Interface for priority queue
type txHeap []*txItem

func (h txHeap) Len() int { return len(h) }

func (h txHeap) Less(i, j int) bool {
	// Higher priority first
	return h[i].priority > h[j].priority
}

func (h txHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
	h[i].index = i
	h[j].index = j
}

func (h *txHeap) Push(x interface{}) {
	n := len(*h)
	item := x.(*txItem)
	item.index = n
	*h = append(*h, item)
}

func (h *txHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	item.index = -1
	*h = old[0 : n-1]
	return item
}
