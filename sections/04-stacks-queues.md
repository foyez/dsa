# 4. Stacks & Queues

## Overview

This section covers stacks (LIFO) and queues (FIFO) with comprehensive examples, templates, and problem-solving patterns.

**Key Topics:**
- Stack fundamentals and implementation
- Monotonic stack pattern
- Queue fundamentals and variants
- Deque and circular queues
- Stack/Queue design problems

---

## 4.1 Stacks

### 4.1.1 Stack Fundamentals

**Definition**: A stack is a linear data structure that follows the **LIFO (Last-In-First-Out)** principle. The last element added to the stack is the first one to be removed.

**Key Operations** (all O(1)):
- `push(item)`: Add to top
- `pop()`: Remove from top  
- `peek()`: View top without removing
- `isEmpty()`: Check if empty

**Real-World**: Stack of plates, browser back button, undo functionality

```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Pop from empty stack")
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("Peek from empty stack")
    
    def is_empty(self):
        return len(self.items) == 0
```

See PATTERN-TEMPLATES.md for complete stack patterns and templates.

---

## 4.2 Queues

### 4.2.1 Queue Fundamentals

**Definition**: A queue is a linear data structure that follows the **FIFO (First-In-First-Out)** principle. The first element added is the first removed.

**Key Operations** (all O(1) with deque):
- `enqueue(item)`: Add to rear
- `dequeue()`: Remove from front
- `front()`: View front without removing

**Real-World**: Line at counter, print queue, BFS traversal

```python
from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        self.items.append(item)
    
    def dequeue(self):
        if not self.is_empty():
            return self.items.popleft()
        raise IndexError("Dequeue from empty queue")
    
    def front(self):
        if not self.is_empty():
            return self.items[0]
        raise IndexError("Front of empty queue")
    
    def is_empty(self):
        return len(self.items) == 0
```

---

## Key Patterns

### Pattern 1: Valid Parentheses (Stack)
```python
def isValid(s):
    """LeetCode 20"""
    stack = []
    pairs = {')':'(', '}':'{', ']':'['}
    
    for char in s:
        if char in pairs.values():
            stack.append(char)
        elif char in pairs:
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()
    
    return len(stack) == 0
```

### Pattern 2: Monotonic Stack
```python
def dailyTemperatures(temperatures):
    """LeetCode 739"""
    result = [0] * len(temperatures)
    stack = []  # indices
    
    for i in range(len(temperatures)):
        while stack and temperatures[stack[-1]] < temperatures[i]:
            prev = stack.pop()
            result[prev] = i - prev
        stack.append(i)
    
    return result
```

### Pattern 3: Min Stack
```python
class MinStack:
    """LeetCode 155"""
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def pop(self):
        val = self.stack.pop()
        if val == self.min_stack[-1]:
            self.min_stack.pop()
    
    def top(self):
        return self.stack[-1]
    
    def getMin(self):
        return self.min_stack[-1]
```

---

## LeetCode Problems (NeetCode.io)

### Stacks - Easy ✅
- 20. Valid Parentheses
- 155. Min Stack  
- 232. Implement Queue using Stacks
- 496. Next Greater Element I

### Stacks - Medium 🟨
- 22. Generate Parentheses
- 71. Simplify Path
- 150. Evaluate Reverse Polish Notation
- 394. Decode String
- 739. Daily Temperatures
- 853. Car Fleet

### Stacks - Hard 🔴
- 84. Largest Rectangle in Histogram
- 224. Basic Calculator
- 316. Remove Duplicate Letters
- 735. Asteroid Collision

### Queues - Easy ✅
- 225. Implement Stack using Queues
- 933. Number of Recent Calls

### Queues - Medium 🟨
- 622. Design Circular Queue
- 641. Design Circular Deque

### Deque - Hard 🔴
- 239. Sliding Window Maximum

---

## Practice Questions

### Fill in the Gaps

1. A stack follows the ________ principle.
2. The time complexity of stack push/pop is ________.
3. A monotonic stack maintains elements in ________ order.
4. A queue follows the ________ principle.
5. Circular queues use ________ arithmetic for wrap-around.

### True or False

1. Stacks allow access from both ends. **[T/F]**
2. Monotonic stacks solve "next greater element" efficiently. **[T/F]**
3. BFS uses a stack for traversal. **[T/F]**
4. Deque allows operations at both ends. **[T/F]**
5. Python's list.pop(0) is O(1). **[T/F]**

### Multiple Choice

1. What's the time complexity of valid parentheses check?
   - A) O(1)
   - B) O(log n)
   - C) O(n)
   - D) O(n²)

2. Which structure is used in BFS?
   - A) Stack
   - B) Queue
   - C) Heap
   - D) Tree

3. Monotonic stack is best for:
   - A) Balancing parentheses
   - B) Next greater element
   - C) Reversing string
   - D) Sorting

---

## Answers

<details>
<summary><strong>View Answers</strong></summary>

### Fill in the Gaps

1. **LIFO (Last-In-First-Out)**
2. **O(1)**
3. **increasing** or **decreasing** (depending on problem)
4. **FIFO (First-In-First-Out)**
5. **modulo**

### True or False

1. **False** - Only from top
2. **True** - O(n) solution
3. **False** - BFS uses queue
4. **True** - Both front and rear
5. **False** - O(n) due to shifting

### Multiple Choice

1. **C** - O(n) single pass
2. **B** - Queue for level-order
3. **B** - Classic use case

</details>

---

*Continue to: [5. Recursion & Backtracking →](05-recursion-backtracking.md)*