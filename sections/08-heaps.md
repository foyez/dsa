# 8. Heaps & Priority Queues

## 8.1 Heap Fundamentals

### 8.1.1 Heap Definition

**Definition**: A heap is a specialized tree-based data structure that satisfies the **heap property**. It's a complete binary tree where each node's value relates to its children's values in a specific way.

**Heap Property:**

1. **Max Heap**: Every parent node ≥ its children
   - Root contains maximum element
   - `parent.val ≥ child.val`

2. **Min Heap**: Every parent node ≤ its children
   - Root contains minimum element
   - `parent.val ≤ child.val`

**Complete Binary Tree**: All levels completely filled except possibly the last, which fills left to right.

**Visual Representation**:
```
Max Heap:               Min Heap:
      50                    1
     /  \                  / \
   30    40               3   2
   / \   /               / \   \
  10 20 35              5   4   6

Properties:
- Root is max/min
- Parent ≥/≤ children
- No order between siblings
- Height: O(log n)
```

**Why Heaps:**
- Get min/max: O(1)
- Insert: O(log n)
- Delete min/max: O(log n)
- Build heap: O(n)
- Space efficient: Array implementation

**Real-World Analogies**:
- **Priority queue**: Hospital ER (highest priority first)
- **Task scheduling**: OS process scheduler
- **Event simulation**: Process events by timestamp
- **Huffman coding**: Build optimal codes

**Array Representation**:
```
Heap:       10
           /  \
          5    3
         / \
        2   4

Array: [10, 5, 3, 2, 4]
Index:  0   1  2  3  4

Parent-Child Relationships:
- Parent of index i: (i-1) // 2
- Left child of i: 2*i + 1
- Right child of i: 2*i + 2

Example:
Index 1 (value 5):
- Parent: (1-1)//2 = 0 (value 10)
- Left child: 2*1+1 = 3 (value 2)
- Right child: 2*1+2 = 4 (value 4)
```

---

### 8.1.2 Heap Operations

#### Heapify (Bubble Down)

**Definition**: Restore heap property by moving element down the tree, swapping with larger/smaller child.

```python
class MaxHeap:
    """Max heap implementation using array"""
    
    def __init__(self):
        self.heap = []
    
    def _parent(self, i):
        """Get parent index"""
        return (i - 1) // 2
    
    def _left_child(self, i):
        """Get left child index"""
        return 2 * i + 1
    
    def _right_child(self, i):
        """Get right child index"""
        return 2 * i + 2
    
    def _heapify_down(self, i):
        """
        Move element down to restore max heap property.
        
        Compare with children, swap with largest if needed.
        Continue until element is larger than both children.
        """
        while True:
            largest = i
            left = self._left_child(i)
            right = self._right_child(i)
            
            # Check if left child is larger
            if left < len(self.heap) and self.heap[left] > self.heap[largest]:
                largest = left
            
            # Check if right child is larger
            if right < len(self.heap) and self.heap[right] > self.heap[largest]:
                largest = right
            
            # If largest is still i, heap property satisfied
            if largest == i:
                break
            
            # Swap and continue
            self.heap[i], self.heap[largest] = self.heap[largest], self.heap[i]
            i = largest
    
    # Time: O(log n) - height of tree
    # Space: O(1)
```

---

#### Heapify Up (Bubble Up)

**Definition**: Restore heap property by moving element up the tree, swapping with parent.

```python
    def _heapify_up(self, i):
        """
        Move element up to restore max heap property.
        
        Compare with parent, swap if larger.
        Continue until element is smaller than parent or becomes root.
        """
        while i > 0:
            parent = self._parent(i)
            
            # If parent is larger, heap property satisfied
            if self.heap[parent] >= self.heap[i]:
                break
            
            # Swap with parent and move up
            self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]
            i = parent
    
    # Time: O(log n)
    # Space: O(1)
```

---

#### Insert

```python
    def insert(self, val):
        """
        Insert new element into heap.
        
        1. Add to end of array (maintains complete tree)
        2. Bubble up to restore heap property
        """
        self.heap.append(val)
        self._heapify_up(len(self.heap) - 1)
    
    # Time: O(log n)
    # Space: O(1)

# Example:
# Insert 15 into [20, 10, 5, 3, 8]
#
# Step 1: Append to end
# [20, 10, 5, 3, 8, 15]
#       20
#      /  \
#    10    5
#   / \   /
#  3   8 15
#
# Step 2: Bubble up (15 > 5, swap)
# [20, 10, 15, 3, 8, 5]
#       20
#      /  \
#    10    15
#   / \   /
#  3   8 5
```

---

#### Extract Max/Min

```python
    def extract_max(self):
        """
        Remove and return maximum element (root).
        
        1. Swap root with last element
        2. Remove last element
        3. Bubble down new root
        """
        if not self.heap:
            return None
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        # Get max (root)
        max_val = self.heap[0]
        
        # Move last element to root
        self.heap[0] = self.heap.pop()
        
        # Restore heap property
        self._heapify_down(0)
        
        return max_val
    
    # Time: O(log n)
    # Space: O(1)

# Example:
# Extract from [20, 10, 15, 3, 8, 5]
#
# Step 1: Swap root with last
# [5, 10, 15, 3, 8, 20]  → pop 20
#
# Step 2: [5, 10, 15, 3, 8]
#       5
#      /  \
#    10    15
#   / \
#  3   8
#
# Step 3: Bubble down 5
# 5 < max(10, 15), swap with 15
# [15, 10, 5, 3, 8]
#       15
#      /  \
#    10    5
#   / \
#  3   8
```

---

#### Peek

```python
    def peek(self):
        """
        Return maximum without removing.
        """
        return self.heap[0] if self.heap else None
    
    # Time: O(1)
    # Space: O(1)
    
    def size(self):
        """Get number of elements"""
        return len(self.heap)
    
    def is_empty(self):
        """Check if heap is empty"""
        return len(self.heap) == 0
```

---

### 8.1.3 Build Heap

**Definition**: Convert an array into a heap in-place.

```python
def build_max_heap(arr):
    """
    Build max heap from unsorted array.
    
    Start from last non-leaf node, heapify down each node.
    Last non-leaf: (n//2 - 1)
    """
    n = len(arr)
    
    # Start from last non-leaf node
    for i in range(n // 2 - 1, -1, -1):
        heapify_down(arr, n, i)
    
    return arr

def heapify_down(arr, n, i):
    """Heapify subtree rooted at index i"""
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
    if left < n and arr[left] > arr[largest]:
        largest = left
    
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify_down(arr, n, largest)

# Time: O(n) - tighter analysis than O(n log n)
# Space: O(1) - in-place

# Example:
# Build heap from [3, 9, 2, 1, 4, 5]
#
# Initial tree:
#       3
#      / \
#     9   2
#    / \ /
#   1  4 5
#
# Heapify from index 1 (value 9):
#       3
#      / \
#     9   2
#    / \ /
#   1  4 5
# 9 > max(1,4), no change
#
# Heapify from index 0 (value 3):
#       9
#      / \
#     4   5
#    / \ /
#   1  3 2
# Result: [9, 4, 5, 1, 3, 2]
```

---

## 8.2 Priority Queue

### 8.2.1 Priority Queue Fundamentals

**Definition**: A priority queue is an abstract data type where each element has a priority. Elements are served in order of priority (highest or lowest first), not FIFO order.

**Operations:**
- `push(item, priority)`: Add element with priority
- `pop()`: Remove element with highest priority
- `peek()`: View highest priority element
- All O(log n) with heap implementation

**Implementation**: Typically implemented using heaps
- Max heap → highest priority first
- Min heap → lowest priority first

**Real-World Uses**:
- **Dijkstra's algorithm**: Shortest path
- **A* search**: Pathfinding
- **Huffman coding**: Compression
- **Event-driven simulation**: Process by time
- **OS task scheduling**: CPU scheduling

---

### 8.2.2 Python's heapq Module

**Definition**: Python's `heapq` module provides min heap implementation.

```python
import heapq

# Min heap operations
heap = []

# Push elements: O(log n)
heapq.heappush(heap, 5)
heapq.heappush(heap, 3)
heapq.heappush(heap, 7)
heapq.heappush(heap, 1)
# heap = [1, 3, 7, 5] (internal min heap structure)

# Pop minimum: O(log n)
min_val = heapq.heappop(heap)  # 1
# heap = [3, 5, 7]

# Peek at minimum: O(1)
min_val = heap[0]  # 3 (don't pop)

# Build heap from list: O(n)
arr = [5, 3, 7, 1]
heapq.heapify(arr)  # Converts to min heap in-place
# arr = [1, 3, 7, 5]

# Push and pop in one operation: O(log n)
heapq.heappushpop(heap, 4)  # Push 4, pop minimum

# Replace root: O(log n)
heapq.heapreplace(heap, 10)  # Pop min, push 10

# N largest/smallest: O(n log k)
nums = [1, 8, 2, 23, 7, -4, 18, 23, 42, 37, 2]
largest_3 = heapq.nlargest(3, nums)    # [42, 37, 23]
smallest_3 = heapq.nsmallest(3, nums)  # [-4, 1, 2]
```

---

### 8.2.3 Max Heap in Python

**Problem**: Python's heapq is min heap. How to get max heap?

**Solution 1**: Negate values

```python
import heapq

# Max heap by negating values
max_heap = []

# Push: negate value
heapq.heappush(max_heap, -5)
heapq.heappush(max_heap, -3)
heapq.heappush(max_heap, -7)
# max_heap = [-7, -5, -3] (largest magnitude first)

# Pop: negate back
max_val = -heapq.heappop(max_heap)  # 7

# Peek: negate
max_val = -max_heap[0]  # 5
```

**Solution 2**: Custom class with comparison

```python
class MaxHeapObj:
    """Wrapper to reverse comparison"""
    def __init__(self, val):
        self.val = val
    
    def __lt__(self, other):
        """Reverse comparison for max heap"""
        return self.val > other.val
    
    def __eq__(self, other):
        return self.val == other.val

# Usage
max_heap = []
heapq.heappush(max_heap, MaxHeapObj(5))
heapq.heappush(max_heap, MaxHeapObj(3))
heapq.heappush(max_heap, MaxHeapObj(7))

max_val = heapq.heappop(max_heap).val  # 7
```

**Solution 3**: Tuple with priority

```python
# Use tuples: (priority, value)
# For max heap, negate priority
max_heap = []

heapq.heappush(max_heap, (-5, "task5"))
heapq.heappush(max_heap, (-3, "task3"))
heapq.heappush(max_heap, (-7, "task7"))

priority, task = heapq.heappop(max_heap)
# priority = -7, task = "task7" (highest priority)
```

---

## 8.3 Heap Problems & Patterns

### 8.3.1 Top K Elements Pattern

**Pattern**: Find K largest/smallest elements from collection.

**Template**:
```python
import heapq

def top_k_template(nums, k):
    """
    Template for top K problems.
    
    For K largest: Use min heap of size K
    For K smallest: Use max heap of size K
    """
    # For K largest:
    heap = []
    
    for num in nums:
        heapq.heappush(heap, num)
        
        # Keep heap size K
        if len(heap) > k:
            heapq.heappop(heap)  # Remove smallest
    
    return heap  # K largest elements

# Time: O(n log k)
# Space: O(k)
```

---

#### Problem: Kth Largest Element

```python
def find_kth_largest(nums, k):
    """
    LeetCode 215: Kth Largest Element in an Array
    
    Find kth largest element.
    
    Approach: Min heap of size k keeps k largest elements.
    Root of heap is kth largest.
    """
    heap = []
    
    for num in nums:
        heapq.heappush(heap, num)
        
        if len(heap) > k:
            heapq.heappop(heap)  # Remove smallest
    
    return heap[0]  # Kth largest

# Time: O(n log k)
# Space: O(k)

# Alternative: Max heap approach
def find_kth_largest_max_heap(nums, k):
    """Using max heap (negate values)"""
    # Negate to create max heap
    max_heap = [-num for num in nums]
    heapq.heapify(max_heap)
    
    # Pop k-1 times
    for _ in range(k - 1):
        heapq.heappop(max_heap)
    
    return -max_heap[0]  # Negate back

# Time: O(n + k log n)
# Space: O(n)

# Example
nums = [3, 2, 1, 5, 6, 4]
k = 2
print(find_kth_largest(nums, k))  # 5
```

---

#### Problem: Top K Frequent Elements

```python
from collections import Counter

def top_k_frequent(nums, k):
    """
    LeetCode 347: Top K Frequent Elements
    
    Return k most frequent elements.
    
    Approach:
    1. Count frequencies
    2. Use min heap of size k with (frequency, num)
    """
    # Count frequencies
    count = Counter(nums)
    
    # Min heap of (frequency, num)
    heap = []
    
    for num, freq in count.items():
        heapq.heappush(heap, (freq, num))
        
        if len(heap) > k:
            heapq.heappop(heap)
    
    # Extract numbers (not frequencies)
    return [num for freq, num in heap]

# Time: O(n log k)
# Space: O(n)

# Alternative: Using nlargest
def top_k_frequent_nlargest(nums, k):
    """Using heapq.nlargest"""
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)

# Example
nums = [1, 1, 1, 2, 2, 3]
k = 2
print(top_k_frequent(nums, k))  # [1, 2]
```

---

#### Problem: K Closest Points to Origin

```python
def k_closest(points, k):
    """
    LeetCode 973: K Closest Points to Origin
    
    Find k closest points to origin (0, 0).
    
    Distance: sqrt(x² + y²)
    Can use x² + y² to avoid sqrt
    """
    # Max heap of (-distance, point)
    # Negate distance for max heap
    heap = []
    
    for x, y in points:
        dist = x*x + y*y
        
        heapq.heappush(heap, (-dist, [x, y]))
        
        if len(heap) > k:
            heapq.heappop(heap)
    
    return [point for dist, point in heap]

# Time: O(n log k)
# Space: O(k)

# Example
points = [[1,3], [-2,2], [5,8], [0,1]]
k = 2
print(k_closest(points, k))  # [[0,1], [-2,2]]
```

---

### 8.3.2 Merge K Sorted Pattern

**Pattern**: Merge multiple sorted structures efficiently.

#### Problem: Merge K Sorted Lists

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_k_lists(lists):
    """
    LeetCode 23: Merge K Sorted Lists
    
    Merge k sorted linked lists into one sorted list.
    
    Approach: Min heap with (value, list_index, node)
    """
    heap = []
    
    # Add first node from each list
    for i, head in enumerate(lists):
        if head:
            heapq.heappush(heap, (head.val, i, head))
    
    dummy = ListNode(0)
    current = dummy
    
    while heap:
        val, i, node = heapq.heappop(heap)
        
        # Add to result
        current.next = node
        current = current.next
        
        # Add next node from same list
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    
    return dummy.next

# Time: O(n log k) where n = total nodes, k = number of lists
# Space: O(k) - heap size

# Example:
# Input: [[1,4,5], [1,3,4], [2,6]]
# Output: [1,1,2,3,4,4,5,6]
```

---

#### Problem: Kth Smallest in Sorted Matrix

```python
def kth_smallest_matrix(matrix, k):
    """
    LeetCode 378: Kth Smallest Element in a Sorted Matrix
    
    n×n matrix where each row and column sorted.
    Find kth smallest element.
    
    Approach: Min heap starting with first column.
    """
    n = len(matrix)
    heap = []
    
    # Add first element from each row
    for r in range(min(n, k)):  # Only need k rows
        heapq.heappush(heap, (matrix[r][0], r, 0))
    
    # Pop k-1 times
    for _ in range(k - 1):
        val, r, c = heapq.heappop(heap)
        
        # Add next element from same row
        if c + 1 < n:
            heapq.heappush(heap, (matrix[r][c + 1], r, c + 1))
    
    return heap[0][0]

# Time: O(k log k) - k operations on heap of size k
# Space: O(k)

# Example
matrix = [
    [1,  5,  9],
    [10, 11, 13],
    [12, 13, 15]
]
k = 8
print(kth_smallest_matrix(matrix, k))  # 13
```

---

### 8.3.3 Running Median Pattern

**Pattern**: Maintain median as elements are added.

#### Problem: Find Median from Data Stream

```python
class MedianFinder:
    """
    LeetCode 295: Find Median from Data Stream
    
    Maintain median of numbers as they're added.
    
    Approach: Two heaps
    - Max heap (left): Smaller half of numbers
    - Min heap (right): Larger half of numbers
    
    Median:
    - If odd count: root of larger heap
    - If even count: average of both roots
    """
    def __init__(self):
        self.small = []  # Max heap (negate values)
        self.large = []  # Min heap
    
    def add_num(self, num):
        """
        Add number to data structure.
        
        Maintain: len(small) == len(large) or len(small) == len(large) + 1
        """
        # Add to max heap (small)
        heapq.heappush(self.small, -num)
        
        # Balance: ensure all in small <= all in large
        if self.small and self.large and (-self.small[0] > self.large[0]):
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        
        # Balance sizes
        if len(self.small) > len(self.large) + 1:
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        
        if len(self.large) > len(self.small):
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -val)
    
    def find_median(self):
        """Return median"""
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2.0

# Time: add_num O(log n), find_median O(1)
# Space: O(n)

# Example
mf = MedianFinder()
mf.add_num(1)    # [1], median = 1
mf.add_num(2)    # [1, 2], median = 1.5
print(mf.find_median())  # 1.5
mf.add_num(3)    # [1, 2, 3], median = 2
print(mf.find_median())  # 2.0

# Visual:
# After adding [1, 2, 3]:
# small (max heap): [2, 1]  ← -2 is root
# large (min heap): [3]
# Median = 2
```

---

### 8.3.4 Scheduling Pattern

#### Problem: Task Scheduler

```python
from collections import Counter

def least_interval(tasks, n):
    """
    LeetCode 621: Task Scheduler
    
    Schedule tasks with cooling period n between same tasks.
    Find minimum intervals needed.
    
    Example: tasks = ["A","A","A","B","B","B"], n = 2
    Output: 8 (A → B → idle → A → B → idle → A → B)
    
    Approach: Greedy with max heap (most frequent first)
    """
    # Count frequencies
    count = Counter(tasks)
    
    # Max heap of frequencies
    max_heap = [-freq for freq in count.values()]
    heapq.heapify(max_heap)
    
    time = 0
    
    while max_heap:
        cycle = []
        
        # Process n+1 tasks in one cycle
        for _ in range(n + 1):
            if max_heap:
                freq = heapq.heappop(max_heap)
                if freq + 1 < 0:  # Still have tasks left
                    cycle.append(freq + 1)
        
        # Add back to heap
        for freq in cycle:
            heapq.heappush(max_heap, freq)
        
        # Add time for this cycle
        if max_heap:
            time += n + 1  # Full cycle
        else:
            time += len(cycle)  # Last cycle
    
    return time

# Time: O(n) - n = number of tasks
# Space: O(1) - at most 26 unique tasks

# Example
tasks = ["A","A","A","B","B","B"]
n = 2
print(least_interval(tasks, n))  # 8
```

---

## 8.4 Heap vs Other Data Structures

| Operation | Heap | BST (balanced) | Sorted Array |
|-----------|------|----------------|--------------|
| **Find min/max** | O(1) | O(log n) | O(1) |
| **Extract min/max** | O(log n) | O(log n) | O(n) |
| **Insert** | O(log n) | O(log n) | O(n) |
| **Build** | O(n) | O(n log n) | O(n log n) |
| **Search arbitrary** | O(n) | O(log n) | O(log n) |
| **Space** | O(n) | O(n) | O(n) |

**When to use Heap:**
- Need quick access to min/max
- Priority queue operations
- Top K elements problems
- Merging sorted structures
- Don't need arbitrary search

**When to use BST:**
- Need ordered traversal
- Range queries
- Arbitrary element search
- Predecessor/successor queries

---

## Practice Questions

### Fill in the Gaps

1. In a max heap, every parent node is ________ than its children.
2. The time complexity to build a heap from an array is ________.
3. Python's heapq module implements a ________ heap.
4. To get kth largest element, use a min heap of size ________.
5. The parent of index i in array representation is at index ________.

### True or False

1. Heaps guarantee sorted order in array representation. **[T/F]**
2. Extract min/max from heap is O(1). **[T/F]**
3. A heap is always a complete binary tree. **[T/F]**
4. Finding an arbitrary element in heap is O(log n). **[T/F]**
5. Heaps can be efficiently implemented using arrays. **[T/F]**

### Multiple Choice

1. Best data structure for "top K elements" problem?
   - A) Array
   - B) Binary Search Tree
   - C) Heap
   - D) Hash Table

2. Time to insert into heap with n elements?
   - A) O(1)
   - B) O(log n)
   - C) O(n)
   - D) O(n log n)

3. For median finding, we use:
   - A) One min heap
   - B) One max heap
   - C) Two heaps
   - D) BST

### Code Challenge

```python
def kth_largest_in_stream(k, nums):
    """
    LeetCode 703: Kth Largest Element in a Stream
    
    Design class that finds kth largest element in stream.
    
    class KthLargest:
        def __init__(self, k, nums):
            # Initialize
            pass
        
        def add(self, val):
            # Add value and return kth largest
            pass
    
    Example:
    kl = KthLargest(3, [4, 5, 8, 2])
    kl.add(3)  # returns 4
    kl.add(5)  # returns 5
    kl.add(10) # returns 5
    kl.add(9)  # returns 8
    
    Implement using heap.
    """
    # Your code here
    pass
```

---

## Answers

<details>
<summary><strong>View Answers</strong></summary>

### Fill in the Gaps

1. **greater than or equal to**
2. **O(n)**
3. **min**
4. **k**
5. **(i - 1) // 2**

### True or False

1. **False** - Heap maintains partial order, not full sort
2. **False** - O(log n) because need to heapify after removal
3. **True** - By definition, heaps are complete binary trees
4. **False** - O(n) because no ordering between siblings
5. **True** - Array representation is standard and efficient

### Multiple Choice

1. **C** - Heap gives O(n log k) solution
2. **B** - O(log n) to bubble up
3. **C** - Max heap for lower half, min heap for upper half

### Code Challenge Answer

```python
import heapq

class KthLargest:
    """
    Maintain min heap of size k with k largest elements.
    Root is kth largest.
    """
    def __init__(self, k, nums):
        self.k = k
        self.heap = nums
        heapq.heapify(self.heap)
        
        # Keep only k largest
        while len(self.heap) > k:
            heapq.heappop(self.heap)
    
    def add(self, val):
        heapq.heappush(self.heap, val)
        
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        
        return self.heap[0]

# Time: __init__ O(n), add O(log k)
# Space: O(k)

# Example
kl = KthLargest(3, [4, 5, 8, 2])
print(kl.add(3))   # 4 - [4, 5, 8]
print(kl.add(5))   # 5 - [5, 5, 8]
print(kl.add(10))  # 5 - [5, 8, 10]
print(kl.add(9))   # 8 - [8, 9, 10]
```

</details>

---

## LeetCode Problems (NeetCode.io)

### Heaps - Easy ✅
- 703. Kth Largest Element in a Stream
- 1046. Last Stone Weight

### Heaps - Medium 🟨
- 215. Kth Largest Element in an Array (IMPORTANT)
- 347. Top K Frequent Elements (IMPORTANT)
- 373. Find K Pairs with Smallest Sums
- 378. Kth Smallest Element in a Sorted Matrix
- 451. Sort Characters By Frequency
- 621. Task Scheduler
- 692. Top K Frequent Words
- 973. K Closest Points to Origin (IMPORTANT)
- 1054. Distant Barcodes

### Heaps - Hard 🔴
- 23. Merge K Sorted Lists (VERY IMPORTANT)
- 295. Find Median from Data Stream (VERY IMPORTANT)
- 502. IPO
- 767. Reorganize String

---

## Summary

### Heap Quick Reference

**When to use heap:**
- Need min/max quickly: O(1)
- Priority queue operations
- Top K elements
- Merge K sorted
- Running median
- Scheduling with priorities

**Key patterns:**
- **Top K**: Min heap of size K for K largest
- **Merge K**: Heap with elements from each structure
- **Running median**: Two heaps (max and min)
- **Scheduling**: Max heap by priority/frequency

**Python heapq tips:**
- Default is min heap
- Max heap: negate values or custom comparator
- Build heap: `heapify()` is O(n)
- Top K: `nlargest()` and `nsmallest()`

---

*Continue to: [9. Hashing →](09-hashing.md)*