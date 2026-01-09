# 3. Linked Lists

## 3.1 Singly Linked Lists

### 3.1.1 Fundamentals

**Definition**: A linked list is a linear data structure where elements (called nodes) are stored in non-contiguous memory locations. Each node contains two parts:
1. **Data**: The value stored in the node
2. **Pointer (or reference)**: Address of the next node in the sequence

Unlike arrays where elements are stored in consecutive memory locations, linked list nodes can be scattered throughout memory, connected by pointers.

**Types of Linked Lists:**
- **Singly Linked List**: Each node points to the next node only
- **Doubly Linked List**: Each node points to both next and previous nodes
- **Circular Linked List**: Last node points back to the first node

**Real-World Analogy**: A linked list is like a treasure hunt where each clue points to the next location. Unlike an array (parking lot with numbered spots), you can only reach spot 5 by following clues from spot 1→2→3→4→5.

**Structure**:
```
[Data|Next] → [Data|Next] → [Data|Next] → None

Order History:
[Order#123|→] → [Order#124|→] → [Order#125|None]
```

**Key Properties**:
- **Dynamic size**: Like adding train cars - can grow/shrink easily
- **No random access**: Must traverse from head to reach middle
- **Efficient insertion/deletion**: O(1) if you have the node reference
- **Sequential access**: O(n) to reach nth element

```python
class OrderNode:
    """
    E-commerce: Order in linked list of customer's order history.
    Each order points to the next older order.
    """
    def __init__(self, order_id, total, product_count):
        self.order_id = order_id
        self.total = total
        self.product_count = product_count
        self.next = None  # Pointer to next order

class OrderHistory:
    """Customer's order history as linked list"""
    def __init__(self):
        self.head = None  # Most recent order
        self.size = 0
    
    def add_order(self, order_id, total, product_count):
        """Add new order at beginning (most recent)"""
        new_order = OrderNode(order_id, total, product_count)
        new_order.next = self.head
        self.head = new_order
        self.size += 1
    
    def get_total_spent(self):
        """Calculate total amount spent - O(n)"""
        total = 0
        current = self.head
        while current:
            total += current.total
            current = current.next
        return total
    
    def find_order(self, order_id):
        """Find specific order - O(n)"""
        current = self.head
        while current:
            if current.order_id == order_id:
                return current
            current = current.next
        return None

# Example usage
history = OrderHistory()
history.add_order("ORD-001", 1999.99, 3)
history.add_order("ORD-002", 49.99, 1)
history.add_order("ORD-003", 299.99, 2)

total = history.get_total_spent()  # 2349.97
order = history.find_order("ORD-002")  # Returns order node
```

### Linked List vs Array Comparison

| Operation | Array | Linked List | Winner |
|-----------|-------|-------------|--------|
| Access by index | O(1) | O(n) | Array |
| Search | O(n) / O(log n) if sorted | O(n) | Array (if sorted) |
| Insert at beginning | O(n) - shift all | O(1) | Linked List |
| Insert at end | O(1)* | O(n) or O(1)** | Depends |
| Insert at middle | O(n) - shift | O(n) - find + O(1) insert | Similar |
| Delete at beginning | O(n) - shift all | O(1) | Linked List |
| Delete at end | O(1) | O(n) without tail pointer | Array |
| Memory | Contiguous | Scattered + pointers | Array |
| Cache performance | Excellent | Poor | Array |

*Amortized for dynamic arrays  
**O(1) if tail pointer maintained

### 3.1.2 Common Operations

#### Reverse a Linked List

```python
def reverse_order_history(head):
    """
    Reverse linked list - make oldest order the head.
    
    Real-world: Display order history oldest-first instead of newest-first.
    
    Technique: Iterative pointer manipulation
    """
    prev = None
    current = head
    
    while current:
        # Save next node
        next_node = current.next
        
        # Reverse the pointer
        current.next = prev
        
        # Move forward
        prev = current
        current = next_node
    
    return prev  # New head (was last node)

# Time: O(n) - Visit each node once
# Space: O(1) - Only 3 pointers

# Visual:
# Before: A → B → C → None
# After:  None ← A ← B ← C
#         ^new head
```

💡 **Memory Trick**: "**P**revious **C**urrent **N**ext" - PCN, like "Piece Cake Now"

#### Detect Cycle (Floyd's Algorithm)

```python
def has_circular_reference(head):
    """
    Detect if linked list has a cycle.
    
    Real-world: Detect circular reference in recommendation system.
    User A recommended by B, B by C, C by A = cycle!
    
    Floyd's Cycle Detection: Tortoise and Hare
    """
    if not head or not head.next:
        return False
    
    slow = head
    fast = head
    
    while fast and fast.next:
        slow = slow.next          # Move 1 step
        fast = fast.next.next     # Move 2 steps
        
        if slow == fast:          # They meet = cycle!
            return True
    
    return False  # Fast reached end = no cycle

# Time: O(n)
# Space: O(1)

# Visual (cycle exists):
# 1 → 2 → 3 → 4
#     ↑       ↓
#     6 ← 5 ←

# Slow: 1 → 2 → 3 → 4 → 5 → 6 → 2 → 3 → 4 (meets fast)
# Fast: 1 → 3 → 5 → 2 → 4 → 6 → 3 → 5 → 2 → 4 (meets slow)
```

#### Find Middle Element

```python
def find_middle_order(head):
    """
    Find middle order in history without counting.
    
    Real-world: Find median customer purchase for analysis.
    
    Technique: Slow/fast pointers
    """
    if not head:
        return None
    
    slow = fast = head
    
    # When fast reaches end, slow is at middle
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow  # Middle node

# Time: O(n)
# Space: O(1)

# Example: 5 orders
# 1 → 2 → 3 → 4 → 5 → None
# slow:    1   2   3 (stops here - middle)
# fast:    1   3   5 (reached end)

# Example: 6 orders (even)
# 1 → 2 → 3 → 4 → 5 → 6 → None
# slow:    1   2   3   4 (second middle)
# fast:    1   3   5   None
```

#### Merge Two Sorted Lists

```python
def merge_sorted_order_lists(list1_head, list2_head):
    """
    Merge two sorted order lists by date.
    
    Real-world: Combine order history from two accounts after merger.
    
    Technique: Two pointers with dummy head
    """
    # Dummy head simplifies edge cases
    dummy = OrderNode(0, 0, 0)
    current = dummy
    
    p1, p2 = list1_head, list2_head
    
    while p1 and p2:
        if p1.order_id < p2.order_id:  # Earlier order first
            current.next = p1
            p1 = p1.next
        else:
            current.next = p2
            p2 = p2.next
        current = current.next
    
    # Attach remaining nodes
    current.next = p1 if p1 else p2
    
    return dummy.next  # Skip dummy head

# Time: O(n + m)
# Space: O(1) - Reusing existing nodes

# Example:
# List 1: 1 → 3 → 5
# List 2: 2 → 4 → 6
# Result: 1 → 2 → 3 → 4 → 5 → 6
```

#### Remove Nth Node From End

```python
def remove_nth_from_end(head, n):
    """
    Remove nth order from the end of history.
    
    Real-world: Delete specific old order (e.g., 3rd from oldest).
    
    Technique: Two pointers with n-gap
    """
    dummy = OrderNode(0, 0, 0)
    dummy.next = head
    
    fast = slow = dummy
    
    # Move fast n+1 steps ahead
    for _ in range(n + 1):
        if fast:
            fast = fast.next
    
    # Move both until fast reaches end
    while fast:
        fast = fast.next
        slow = slow.next
    
    # Remove node
    slow.next = slow.next.next
    
    return dummy.next

# Time: O(L) where L = list length
# Space: O(1)

# Example: Remove 2nd from end in [1,2,3,4,5], n=2
# Want to remove 4:
# 1 → 2 → 3 → 4 → 5 → None
#         ↑slow  ↑fast (gap of 2)
# When fast reaches None, slow is at node before target
```

---

## 3.2 Doubly & Circular Lists

### 3.2.1 Doubly Linked Lists

**Definition**: A doubly linked list is a linked list where each node contains three parts:
1. **Data**: The value stored
2. **Next pointer**: Reference to the next node
3. **Previous pointer**: Reference to the previous node

This bidirectional linking allows traversal in both forward and backward directions.

**Advantages over Singly Linked Lists:**
- Can traverse backward without recursion
- Easier deletion (no need to track previous node)
- Can insert before a given node efficiently

**Disadvantages:**
- More memory per node (extra pointer)
- More complex implementation

**Real-World Analogy**: Browser history - go back (prev) and forward (next)

```python
class BrowserHistoryNode:
    """
    Doubly linked list node for browser history.
    Can navigate both backward and forward.
    """
    def __init__(self, url):
        self.url = url
        self.prev = None  # Previous page
        self.next = None  # Next page

class BrowserHistory:
    """
    Browser with back/forward navigation.
    
    Real-world: Chrome/Firefox history navigation.
    """
    def __init__(self, homepage):
        self.current = BrowserHistoryNode(homepage)
    
    def visit(self, url):
        """Visit new page - clears forward history"""
        new_page = BrowserHistoryNode(url)
        new_page.prev = self.current
        self.current.next = new_page
        self.current = new_page
    
    def back(self, steps):
        """Go back in history"""
        while steps > 0 and self.current.prev:
            self.current = self.current.prev
            steps -= 1
        return self.current.url
    
    def forward(self, steps):
        """Go forward in history"""
        while steps > 0 and self.current.next:
            self.current = self.current.next
            steps -= 1
        return self.current.url

# Time: O(1) per operation
# Space: O(n) for n visited pages

# Example usage
browser = BrowserHistory("google.com")
browser.visit("amazon.com")
browser.visit("facebook.com")
url = browser.back(1)        # amazon.com
url = browser.forward(1)     # facebook.com
```

### 3.2.2 Circular Linked Lists

**Definition**: A circular linked list is a linked list where the last node points back to the first node instead of pointing to null. This creates a circular chain where you can traverse the entire list starting from any node.

**Key Characteristics:**
- No null pointer at the end
- Can start traversal from any node
- Useful for round-robin scheduling and circular buffers
- Can be singly or doubly linked

**How to identify the end:** 
- In singly circular: when `current.next == head`
- In doubly circular: when `current.next == head` or `current.prev == tail`

**Real-World Analogy**: Music playlist on repeat - last song links back to first

```python
class SongNode:
    """Node in circular playlist"""
    def __init__(self, title, artist):
        self.title = title
        self.artist = artist
        self.next = None

class CircularPlaylist:
    """
    Circular linked list for music playlist on repeat.
    
    Real-world: Spotify/Apple Music repeat mode.
    """
    def __init__(self):
        self.head = None
        self.size = 0
    
    def add_song(self, title, artist):
        """Add song to end of playlist"""
        new_song = SongNode(title, artist)
        
        if not self.head:
            self.head = new_song
            new_song.next = new_song  # Point to itself
        else:
            # Find last song
            current = self.head
            while current.next != self.head:
                current = current.next
            
            current.next = new_song
            new_song.next = self.head  # Complete the circle
        
        self.size += 1
    
    def play_n_songs(self, n, start_song=None):
        """
        Play n songs starting from current (or specified) song.
        Automatically loops back to beginning.
        """
        if not self.head:
            return []
        
        current = start_song if start_song else self.head
        playlist = []
        
        for _ in range(n):
            playlist.append(f"{current.title} - {current.artist}")
            current = current.next
        
        return playlist

# Example
playlist = CircularPlaylist()
playlist.add_song("Song A", "Artist 1")
playlist.add_song("Song B", "Artist 2")
playlist.add_song("Song C", "Artist 3")

# Play 5 songs (will loop after C)
songs = playlist.play_n_songs(5)
# ['Song A', 'Song B', 'Song C', 'Song A', 'Song B']
```

---

## Practice Questions - Linked Lists

### Fill in the Gaps

1. Reversing a linked list iteratively requires ________ pointers: previous, current, and next.
2. Floyd's cycle detection uses ________ and ________ pointers.
3. To find the middle of a linked list, when the fast pointer reaches the end, the ________ pointer is at the middle.
4. Doubly linked lists have ________ pointers per node compared to singly linked lists.
5. The time complexity of accessing the nth element in a linked list is ________.

### True or False

1. Linked lists have better cache performance than arrays. **[T/F]**
2. Inserting at the beginning of a linked list is O(1). **[T/F]**
3. You need extra space proportional to n to reverse a linked list. **[T/F]**
4. Finding the middle element requires knowing the list length first. **[T/F]**
5. Doubly linked lists use twice the memory of singly linked lists. **[T/F]**

### Multiple Choice

1. What is the time complexity of reversing a linked list?
   - A) O(1)
   - B) O(log n)
   - C) O(n)
   - D) O(n²)

2. To detect a cycle, what does it mean when fast and slow pointers meet?
   - A) The list is sorted
   - B) A cycle exists
   - C) The list length is even
   - D) Middle element found

3. Which operation is faster in a linked list than an array?
   - A) Access by index
   - B) Search for element
   - C) Insert at beginning
   - D) Find maximum element

### Code Challenge

```python
def remove_duplicates_from_sorted_list(head):
    """
    Remove duplicates from sorted linked list.
    Example: 1→1→2→3→3 becomes 1→2→3
    """
    # Your code here
    pass
```

---

## Answers - Linked Lists

<details>
<summary><strong>View Answers</strong></summary>

### Fill in the Gaps
1. three
2. slow, fast (or tortoise, hare)
3. slow
4. two (prev and next vs just next)
5. O(n)

### True or False
1. **False** - Arrays have better cache performance (contiguous memory)
2. **True** - Just update head pointer
3. **False** - Can reverse in O(1) space iteratively
4. **False** - Use slow/fast pointers without counting
5. **False** - Not exactly twice (approximately ~1.5x due to extra pointer per node)

### Multiple Choice
1. **C** - O(n), must visit each node
2. **B** - Meeting indicates cycle exists
3. **C** - Insert at beginning is O(1) for list vs O(n) for array

### Code Challenge Answer
```python
def remove_duplicates_from_sorted_list(head):
    """Remove consecutive duplicates"""
    if not head:
        return None
    
    current = head
    
    while current and current.next:
        if current.order_id == current.next.order_id:
            current.next = current.next.next  # Skip duplicate
        else:
            current = current.next
    
    return head

# Time: O(n)
# Space: O(1)
```

</details>

---

## LeetCode Problems - Linked Lists

### Easy
- ✅ 21. Merge Two Sorted Lists
- ✅ 83. Remove Duplicates from Sorted List
- ✅ 141. Linked List Cycle
- ✅ 160. Intersection of Two Linked Lists
- ✅ 203. Remove Linked List Elements
- ✅ 206. Reverse Linked List
- ✅ 234. Palindrome Linked List
- ✅ 237. Delete Node in a Linked List
- ✅ 876. Middle of the Linked List

### Medium
- 🟨 2. Add Two Numbers
- 🟨 19. Remove Nth Node From End of List
- 🟨 24. Swap Nodes in Pairs
- 🟨 61. Rotate List
- 🟨 82. Remove Duplicates from Sorted List II
- 🟨 86. Partition List
- 🟨 92. Reverse Linked List II
- 🟨 138. Copy List with Random Pointer
- 🟨 142. Linked List Cycle II
- 🟨 143. Reorder List
- 🟨 146. LRU Cache
- 🟨 148. Sort List
- 🟨 328. Odd Even Linked List
- 🟨 430. Flatten a Multilevel Doubly Linked List

### Hard
- 🔴 23. Merge k Sorted Lists
- 🔴 25. Reverse Nodes in k-Group

---

*Continue to: [4. Stacks & Queues →](04-stacks-queues.md)*