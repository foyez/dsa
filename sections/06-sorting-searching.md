# 6. Sorting & Searching

## 6.1 Sorting Algorithms

### 6.1.1 Sorting Fundamentals

**Definition**: Sorting is the process of arranging elements in a specific order (ascending or descending) based on a comparison criterion. Sorting algorithms are fundamental to computer science and are used to optimize searching, data organization, and problem-solving.

**Why Sorting Matters:**
- Enables binary search (O(log n) vs O(n))
- Simplifies finding duplicates, median, mode
- Required for many algorithms (merge intervals, two pointers)
- Database indexing and query optimization
- Real-world: Organizing files, ranking results, scheduling

**Key Characteristics:**
- **Time Complexity**: How fast it sorts n elements
- **Space Complexity**: Extra memory needed
- **Stability**: Preserves relative order of equal elements
- **In-place**: Uses O(1) extra space
- **Comparison-based**: Uses comparisons to determine order

**Stable vs Unstable**:
```
Input: [(3, "a"), (1, "b"), (3, "c"), (2, "d")]
Sorting by first element:

Stable:   [(1, "b"), (2, "d"), (3, "a"), (3, "c")]
          ↑ Original order of (3,"a") and (3,"c") preserved

Unstable: [(1, "b"), (2, "d"), (3, "c"), (3, "a")]
          ↑ Order of equal elements may change
```

**Real-World Analogies**:
- **Sorting cards**: Various strategies (insertion, merge, quick)
- **Library organization**: Books sorted by author, then title
- **Leaderboard**: Scores sorted high to low
- **File explorer**: Files sorted by name, date, or size

---

### 6.1.2 Comparison-Based Sorts

#### Bubble Sort

**Definition**: Repeatedly steps through the list, compares adjacent elements and swaps them if they're in wrong order. Largest element "bubbles up" to the end each pass.

**How It Works:**
```
Pass 1: [5, 2, 8, 1] → [2, 5, 1, 8]  (8 bubbles to end)
Pass 2: [2, 5, 1, 8] → [2, 1, 5, 8]  (5 bubbles)
Pass 3: [2, 1, 5, 8] → [1, 2, 5, 8]  (2 bubbles)
Done!
```

**Implementation**:
```python
def bubble_sort(arr):
    """
    Sort array using bubble sort.
    
    Compares adjacent elements, swaps if out of order.
    Each pass moves largest unsorted element to its position.
    
    Stable: Yes
    In-place: Yes
    """
    n = len(arr)
    
    for i in range(n):
        # Flag to optimize: stop if no swaps made
        swapped = False
        
        # Last i elements already in place
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        
        # If no swaps, array is sorted
        if not swapped:
            break
    
    return arr

# Time: O(n²) worst/average, O(n) best (already sorted)
# Space: O(1)

# Example
arr = [64, 34, 25, 12, 22, 11, 90]
print(bubble_sort(arr))  # [11, 12, 22, 25, 34, 64, 90]
```

**When to Use**: Educational purposes, tiny arrays, nearly sorted data

---

#### Selection Sort

**Definition**: Divides array into sorted and unsorted portions. Repeatedly selects smallest element from unsorted portion and moves it to sorted portion.

**How It Works:**
```
[64, 25, 12, 22, 11]
 ↓ Find min (11), swap with first
[11, 25, 12, 22, 64]
     ↓ Find min in rest (12), swap
[11, 12, 25, 22, 64]
         ↓ Find min (22), swap
[11, 12, 22, 25, 64]
             ↓ Already sorted
```

**Implementation**:
```python
def selection_sort(arr):
    """
    Sort by repeatedly selecting minimum element.
    
    Finds minimum in unsorted portion, swaps to front.
    
    Stable: No (can be made stable with careful swapping)
    In-place: Yes
    """
    n = len(arr)
    
    for i in range(n):
        # Find minimum in unsorted portion
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        
        # Swap minimum to current position
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    
    return arr

# Time: O(n²) in all cases
# Space: O(1)

# Example
arr = [64, 25, 12, 22, 11]
print(selection_sort(arr))  # [11, 12, 22, 25, 64]
```

**When to Use**: Memory writes are expensive (swaps minimal), simple implementation needed

---

#### Insertion Sort

**Definition**: Builds sorted array one element at a time by inserting each element into its correct position among previously sorted elements.

**How It Works:**
```
[5, 2, 4, 6, 1, 3]
[5] [2, 4, 6, 1, 3]  ← Start with first element sorted
[2, 5] [4, 6, 1, 3]  ← Insert 2
[2, 4, 5] [6, 1, 3]  ← Insert 4
[2, 4, 5, 6] [1, 3]  ← Insert 6
[1, 2, 4, 5, 6] [3]  ← Insert 1
[1, 2, 3, 4, 5, 6]   ← Insert 3
```

**Implementation**:
```python
def insertion_sort(arr):
    """
    Sort by inserting elements into sorted portion.
    
    Like sorting playing cards: pick card, insert into sorted hand.
    
    Stable: Yes
    In-place: Yes
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        
        # Shift elements greater than key to the right
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        
        # Insert key at correct position
        arr[j + 1] = key
    
    return arr

# Time: O(n²) worst/average, O(n) best (already sorted)
# Space: O(1)

# Example
arr = [12, 11, 13, 5, 6]
print(insertion_sort(arr))  # [5, 6, 11, 12, 13]
```

**When to Use**: Small arrays, nearly sorted data, online sorting (elements arrive one by one)

---

#### Merge Sort

**Definition**: Divide and conquer algorithm that divides array into halves, recursively sorts them, then merges sorted halves back together.

**How It Works:**
```
[38, 27, 43, 3, 9, 82, 10]

Divide:
[38, 27, 43, 3] [9, 82, 10]
[38, 27] [43, 3] [9, 82] [10]
[38] [27] [43] [3] [9] [82] [10]

Merge:
[27, 38] [3, 43] [9, 82] [10]
[3, 27, 38, 43] [9, 10, 82]
[3, 9, 10, 27, 38, 43, 82]
```

**Implementation**:
```python
def merge_sort(arr):
    """
    Sort using divide and conquer.
    
    Divides array in half, recursively sorts halves,
    then merges sorted halves.
    
    Stable: Yes
    In-place: No (requires extra space for merging)
    """
    # Base case: array of size 0 or 1
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]
    
    # Conquer (recursively sort)
    left_sorted = merge_sort(left)
    right_sorted = merge_sort(right)
    
    # Combine (merge)
    return merge(left_sorted, right_sorted)

def merge(left, right):
    """Merge two sorted arrays into one sorted array"""
    result = []
    i = j = 0
    
    # Compare elements from both arrays
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Add remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result

# Time: O(n log n) in all cases
# Space: O(n) - temporary arrays for merging

# Example
arr = [38, 27, 43, 3, 9, 82, 10]
print(merge_sort(arr))  # [3, 9, 10, 27, 38, 43, 82]
```

**When to Use**: Guaranteed O(n log n), need stability, linked lists, external sorting

---

#### Quick Sort

**Definition**: Picks a pivot element, partitions array so elements smaller than pivot are on left and larger on right, then recursively sorts partitions.

**How It Works:**
```
[10, 7, 8, 9, 1, 5]  pivot=5 (last element)

Partition:
[1] [5] [7, 8, 9, 10]
     ↑ Pivot in final position

Recursively sort:
Left: [1] - already sorted
Right: [7, 8, 9, 10]  pivot=10
       [7, 8, 9] [10]
       ...

Result: [1, 5, 7, 8, 9, 10]
```

**Implementation**:
```python
def quick_sort(arr, low=0, high=None):
    """
    Sort using quick sort (in-place).
    
    Picks pivot, partitions around it, recursively sorts partitions.
    
    Stable: No
    In-place: Yes
    """
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        # Partition and get pivot index
        pivot_idx = partition(arr, low, high)
        
        # Recursively sort partitions
        quick_sort(arr, low, pivot_idx - 1)
        quick_sort(arr, pivot_idx + 1, high)
    
    return arr

def partition(arr, low, high):
    """
    Partition array around pivot (last element).
    Returns final position of pivot.
    """
    pivot = arr[high]
    i = low - 1  # Index of smaller element
    
    for j in range(low, high):
        # If current element smaller than pivot
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    # Place pivot in correct position
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# Time: O(n log n) average, O(n²) worst case
# Space: O(log n) - recursion stack

# Example
arr = [10, 7, 8, 9, 1, 5]
print(quick_sort(arr))  # [1, 5, 7, 8, 9, 10]
```

**Optimization: Random Pivot**
```python
import random

def quick_sort_randomized(arr, low=0, high=None):
    """Quick sort with random pivot to avoid worst case"""
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        # Randomly choose pivot
        pivot_idx = random.randint(low, high)
        arr[pivot_idx], arr[high] = arr[high], arr[pivot_idx]
        
        pivot_idx = partition(arr, low, high)
        quick_sort_randomized(arr, low, pivot_idx - 1)
        quick_sort_randomized(arr, pivot_idx + 1, high)
    
    return arr

# Time: O(n log n) expected, O(n²) worst (very rare)
```

**When to Use**: General purpose, cache-friendly, average case performance critical

---

#### Heap Sort

**Definition**: Builds max heap from array, repeatedly extracts maximum element and places it at end of array.

**Implementation**:
```python
def heap_sort(arr):
    """
    Sort using heap data structure.
    
    1. Build max heap
    2. Repeatedly extract max and rebuild heap
    
    Stable: No
    In-place: Yes
    """
    n = len(arr)
    
    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    
    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        # Move current root to end
        arr[0], arr[i] = arr[i], arr[0]
        
        # Heapify reduced heap
        heapify(arr, i, 0)
    
    return arr

def heapify(arr, n, i):
    """
    Maintain max heap property for subtree rooted at i.
    n is size of heap.
    """
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
    # Check if left child is larger
    if left < n and arr[left] > arr[largest]:
        largest = left
    
    # Check if right child is larger
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    # If largest is not root, swap and heapify
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

# Time: O(n log n) in all cases
# Space: O(1)

# Example
arr = [12, 11, 13, 5, 6, 7]
print(heap_sort(arr))  # [5, 6, 7, 11, 12, 13]
```

**When to Use**: Guaranteed O(n log n), memory constrained, selection algorithms (top K)

---

### 6.1.3 Non-Comparison Sorts

#### Counting Sort

**Definition**: Counts occurrences of each value, then reconstructs sorted array. Only works for non-negative integers with limited range.

**Implementation**:
```python
def counting_sort(arr):
    """
    Sort by counting occurrences.
    
    Works when elements are in range [0, k] where k is not too large.
    
    Stable: Yes (can be implemented stable)
    In-place: No
    """
    if not arr:
        return arr
    
    # Find range
    max_val = max(arr)
    min_val = min(arr)
    range_size = max_val - min_val + 1
    
    # Count occurrences
    count = [0] * range_size
    for num in arr:
        count[num - min_val] += 1
    
    # Reconstruct sorted array
    result = []
    for i in range(range_size):
        result.extend([i + min_val] * count[i])
    
    return result

# Time: O(n + k) where k = range of values
# Space: O(k)

# Example
arr = [4, 2, 2, 8, 3, 3, 1]
print(counting_sort(arr))  # [1, 2, 2, 3, 3, 4, 8]
```

**When to Use**: Small range of integers, need linear time, duplicate values

---

#### Radix Sort

**Definition**: Sorts by processing digits from least significant to most significant. Uses stable sort (like counting sort) for each digit.

**Implementation**:
```python
def radix_sort(arr):
    """
    Sort non-negative integers by processing digits.
    
    Sorts digit by digit from rightmost to leftmost.
    
    Stable: Yes
    In-place: No
    """
    if not arr:
        return arr
    
    # Find maximum to determine number of digits
    max_val = max(arr)
    
    # Process each digit position
    exp = 1  # 10^0, 10^1, 10^2, ...
    while max_val // exp > 0:
        counting_sort_by_digit(arr, exp)
        exp *= 10
    
    return arr

def counting_sort_by_digit(arr, exp):
    """Stable counting sort by specific digit position"""
    n = len(arr)
    output = [0] * n
    count = [0] * 10  # Digits 0-9
    
    # Count occurrences of each digit
    for num in arr:
        digit = (num // exp) % 10
        count[digit] += 1
    
    # Convert to actual positions
    for i in range(1, 10):
        count[i] += count[i - 1]
    
    # Build output array (stable)
    for i in range(n - 1, -1, -1):
        digit = (arr[i] // exp) % 10
        output[count[digit] - 1] = arr[i]
        count[digit] -= 1
    
    # Copy to original array
    for i in range(n):
        arr[i] = output[i]

# Time: O(d × (n + k)) where d = digits, k = base (10)
# Space: O(n + k)

# Example
arr = [170, 45, 75, 90, 802, 24, 2, 66]
print(radix_sort(arr))  # [2, 24, 45, 66, 75, 90, 170, 802]
```

**When to Use**: Large numbers of integers, fixed number of digits, need linear time

---

### 6.1.4 Sorting Algorithm Comparison

| Algorithm | Best | Average | Worst | Space | Stable | In-place |
|-----------|------|---------|-------|-------|--------|----------|
| **Bubble** | O(n) | O(n²) | O(n²) | O(1) | Yes | Yes |
| **Selection** | O(n²) | O(n²) | O(n²) | O(1) | No* | Yes |
| **Insertion** | O(n) | O(n²) | O(n²) | O(1) | Yes | Yes |
| **Merge** | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes | No |
| **Quick** | O(n log n) | O(n log n) | O(n²) | O(log n) | No | Yes |
| **Heap** | O(n log n) | O(n log n) | O(n log n) | O(1) | No | Yes |
| **Counting** | O(n+k) | O(n+k) | O(n+k) | O(k) | Yes | No |
| **Radix** | O(d(n+k)) | O(d(n+k)) | O(d(n+k)) | O(n+k) | Yes | No |

*Can be made stable

**Which to Use When:**
- **Small arrays (n < 10)**: Insertion sort
- **Nearly sorted**: Insertion sort, Bubble sort
- **Need stability**: Merge sort, Insertion sort
- **Limited memory**: Heap sort, Quick sort (in-place)
- **General purpose**: Quick sort, Merge sort
- **Guaranteed O(n log n)**: Merge sort, Heap sort
- **Integers in range**: Counting sort, Radix sort

---

### 6.1.5 Custom Sorting

**Python's sorted() and list.sort()**:
```python
# sorted() - returns new sorted list
arr = [3, 1, 4, 1, 5]
sorted_arr = sorted(arr)  # [1, 1, 3, 4, 5]

# list.sort() - sorts in place
arr.sort()  # arr is now [1, 1, 3, 4, 5]

# Both use Timsort: hybrid of merge sort and insertion sort
# Time: O(n log n), Stable: Yes
```

**Custom Comparisons**:
```python
# Sort by custom key
students = [
    {'name': 'Alice', 'grade': 85},
    {'name': 'Bob', 'grade': 92},
    {'name': 'Charlie', 'grade': 78}
]

# Sort by grade (ascending)
sorted_students = sorted(students, key=lambda x: x['grade'])
# [{'name': 'Charlie', 'grade': 78}, ...]

# Sort by grade (descending)
sorted_students = sorted(students, key=lambda x: x['grade'], reverse=True)
# [{'name': 'Bob', 'grade': 92}, ...]

# Sort by multiple criteria: grade desc, then name asc
sorted_students = sorted(students, key=lambda x: (-x['grade'], x['name']))

# Sort by length of string
words = ['apple', 'pie', 'banana', 'cat']
sorted(words, key=len)  # ['pie', 'cat', 'apple', 'banana']
```

**LeetCode Example: Sort Colors (Dutch National Flag)**
```python
def sort_colors(nums):
    """
    LeetCode 75: Sort Colors
    
    Sort array with values 0, 1, 2 in-place.
    
    Dutch National Flag algorithm: three-way partitioning
    """
    low = 0      # Boundary for 0s
    mid = 0      # Current element
    high = len(nums) - 1  # Boundary for 2s
    
    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:  # nums[mid] == 2
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
    
    return nums

# Time: O(n) - single pass
# Space: O(1)

# Example
nums = [2, 0, 2, 1, 1, 0]
print(sort_colors(nums))  # [0, 0, 1, 1, 2, 2]
```

---

## 6.2 Binary Search

### 6.2.1 Binary Search Fundamentals

**Definition**: Binary search is an efficient algorithm for finding a target value in a **sorted array** by repeatedly dividing the search interval in half. If the target is less than the middle element, search the left half; if greater, search the right half.

**Key Requirements:**
- Array must be sorted
- Random access (array, not linked list)
- Comparison operation available

**Why Binary Search:**
- O(log n) time vs O(n) linear search
- Critical for large datasets
- Foundation for many advanced algorithms

**How It Works:**
```
Search for 7 in [1, 3, 5, 7, 9, 11, 13]

Step 1: left=0, right=6, mid=3
        [1, 3, 5, 7, 9, 11, 13]
                 ↑
        7 == 7? Yes! Found at index 3

If searching for 9:
Step 1: mid=3, arr[3]=7, 9 > 7
        Search right: [9, 11, 13]
Step 2: mid=5, arr[5]=11, 9 < 11
        Search left: [9]
Step 3: mid=4, arr[4]=9, Found!
```

**Real-World Analogies:**
- **Dictionary**: Open to middle, go left/right based on word
- **Phone book**: Find name by halving pages
- **Guessing game**: "Higher/Lower" - halve range each guess
- **Library**: Find book by section, then subsection, etc.

---

### 6.2.2 Binary Search Templates

#### Template 1: Standard Binary Search

```python
def binary_search(arr, target):
    """
    Standard binary search template.
    
    Find exact target in sorted array.
    Returns index if found, -1 otherwise.
    
    Use when: Finding exact match in sorted array
    """
    left = 0
    right = len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2  # Avoid overflow
        
        if arr[mid] == target:
            return mid  # Found
        elif arr[mid] < target:
            left = mid + 1  # Search right
        else:
            right = mid - 1  # Search left
    
    return -1  # Not found

# Time: O(log n)
# Space: O(1)

# Example
arr = [1, 3, 5, 7, 9, 11, 13, 15]
print(binary_search(arr, 7))   # 3
print(binary_search(arr, 10))  # -1
```

**Key Points:**
- `left <= right` (inclusive)
- `mid = left + (right - left) // 2` to prevent overflow
- Return -1 if not found

---

#### Template 2: Leftmost Binary Search

```python
def binary_search_leftmost(arr, target):
    """
    Find leftmost (first) occurrence of target.
    
    If target not present, returns insertion position.
    
    Use when: Finding first occurrence, insertion point
    """
    left = 0
    right = len(arr)  # Note: not len(arr) - 1
    
    while left < right:  # Note: not <=
        mid = left + (right - left) // 2
        
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid  # Don't exclude mid
    
    return left  # Leftmost position

# Time: O(log n)
# Space: O(1)

# Example
arr = [1, 2, 2, 2, 3, 4, 5]
print(binary_search_leftmost(arr, 2))  # 1 (first occurrence)
print(binary_search_leftmost(arr, 6))  # 7 (insertion point)
```

**Use Cases:**
- `bisect.bisect_left()` in Python
- Finding insertion position
- Lower bound queries

---

#### Template 3: Rightmost Binary Search

```python
def binary_search_rightmost(arr, target):
    """
    Find rightmost (last) occurrence of target.
    
    Returns position after last occurrence.
    
    Use when: Finding last occurrence, upper bound
    """
    left = 0
    right = len(arr)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if arr[mid] <= target:  # Note: <=
            left = mid + 1
        else:
            right = mid
    
    return left  # Position after last occurrence

# Time: O(log n)
# Space: O(1)

# Example
arr = [1, 2, 2, 2, 3, 4, 5]
print(binary_search_rightmost(arr, 2))  # 4 (position after last 2)
print(binary_search_rightmost(arr, 2) - 1)  # 3 (last occurrence)
```

**Use Cases:**
- `bisect.bisect_right()` in Python
- Finding upper bound
- Range queries

---

#### Template 4: Binary Search on Answer

```python
def binary_search_on_answer(condition, low, high):
    """
    Binary search on answer space.
    
    Find minimum/maximum value that satisfies condition.
    
    Use when: Optimizing a value, "minimum capacity", "maximum speed"
    """
    result = -1
    
    while low <= high:
        mid = low + (high - low) // 2
        
        if is_valid(mid):  # Check if mid works
            result = mid  # Save this answer
            high = mid - 1  # Try to minimize (or low = mid + 1 to maximize)
        else:
            low = mid + 1  # Need larger value
    
    return result

def is_valid(capacity):
    """Problem-specific validation function"""
    # Check if given capacity/speed/value works
    pass

# Time: O(log(range) × validation_time)
# Space: O(1)
```

**Pattern Recognition:**
- "Minimum capacity to ship packages"
- "Maximum speed to eat bananas"
- "Smallest divisor"
- "Split array largest sum"

---

### 6.2.3 Common Binary Search Problems

#### Problem: Search in Rotated Sorted Array

```python
def search_rotated(nums, target):
    """
    LeetCode 33: Search in Rotated Sorted Array
    
    Array sorted but rotated at pivot point.
    Example: [4,5,6,7,0,1,2] rotated from [0,1,2,4,5,6,7]
    
    Key insight: One half is always sorted.
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # Determine which half is sorted
        if nums[left] <= nums[mid]:
            # Left half is sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1  # Target in left half
            else:
                left = mid + 1   # Target in right half
        else:
            # Right half is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1   # Target in right half
            else:
                right = mid - 1  # Target in left half
    
    return -1

# Time: O(log n)
# Space: O(1)

# Example
nums = [4, 5, 6, 7, 0, 1, 2]
print(search_rotated(nums, 0))  # 4
print(search_rotated(nums, 3))  # -1
```

---

#### Problem: Find Minimum in Rotated Array

```python
def find_min_rotated(nums):
    """
    LeetCode 153: Find Minimum in Rotated Sorted Array
    
    Find minimum element in rotated sorted array.
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] > nums[right]:
            # Minimum is in right half
            left = mid + 1
        else:
            # Minimum is in left half or mid
            right = mid
    
    return nums[left]

# Time: O(log n)
# Space: O(1)

# Example
print(find_min_rotated([3, 4, 5, 1, 2]))  # 1
print(find_min_rotated([4, 5, 6, 7, 0, 1, 2]))  # 0
```

---

#### Problem: Search 2D Matrix

```python
def search_matrix(matrix, target):
    """
    LeetCode 74: Search a 2D Matrix
    
    Search in matrix where:
    - Each row sorted left to right
    - First element of each row > last element of previous row
    
    Treat as 1D sorted array!
    """
    if not matrix or not matrix[0]:
        return False
    
    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        # Convert 1D index to 2D
        row = mid // n
        col = mid % n
        mid_val = matrix[row][col]
        
        if mid_val == target:
            return True
        elif mid_val < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return False

# Time: O(log(m × n))
# Space: O(1)

# Example
matrix = [
    [1,  3,  5,  7],
    [10, 11, 16, 20],
    [23, 30, 34, 60]
]
print(search_matrix(matrix, 3))   # True
print(search_matrix(matrix, 13))  # False
```

---

#### Problem: First Bad Version

```python
def first_bad_version(n):
    """
    LeetCode 278: First Bad Version
    
    Find first bad version using minimum API calls.
    
    Given: isBadVersion(version) returns True if bad
    
    Template: Leftmost binary search
    """
    left, right = 1, n
    
    while left < right:
        mid = left + (right - left) // 2
        
        if isBadVersion(mid):
            right = mid  # First bad is mid or earlier
        else:
            left = mid + 1  # First bad is after mid
    
    return left

# Time: O(log n)
# Space: O(1)

# Example (assuming isBadVersion is defined):
# Versions: [good, good, bad, bad, bad]
# Returns: 3 (first bad version)
```

---

#### Problem: Koko Eating Bananas

```python
def min_eating_speed(piles, h):
    """
    LeetCode 875: Koko Eating Bananas
    
    Find minimum eating speed k such that Koko can eat all bananas in h hours.
    Each hour, she eats at most k bananas from one pile.
    
    Binary search on answer: k ranges from 1 to max(piles)
    """
    def can_finish(k):
        """Check if can finish all piles with speed k in h hours"""
        hours = 0
        for pile in piles:
            hours += (pile + k - 1) // k  # Ceiling division
        return hours <= h
    
    left, right = 1, max(piles)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if can_finish(mid):
            right = mid  # Try smaller speed
        else:
            left = mid + 1  # Need faster speed
    
    return left

# Time: O(n log m) where m = max(piles)
# Space: O(1)

# Example
piles = [3, 6, 7, 11]
h = 8
print(min_eating_speed(piles, h))  # 4
# Hour 1: eat 3, Hour 2: eat 4 from 6, Hour 3: eat 2 from 6,
# Hour 4: eat 4 from 7, Hour 5: eat 3 from 7, 
# Hours 6-8: eat 11 (4+4+3)
```

---

#### Problem: Capacity To Ship Packages

```python
def ship_within_days(weights, days):
    """
    LeetCode 1011: Capacity To Ship Packages Within D Days
    
    Find minimum ship capacity to ship all packages in D days.
    Packages must be shipped in order.
    
    Binary search on answer: capacity ranges from max(weights) to sum(weights)
    """
    def can_ship(capacity):
        """Check if can ship with given capacity in days"""
        current_load = 0
        days_needed = 1
        
        for weight in weights:
            if current_load + weight > capacity:
                days_needed += 1
                current_load = weight
            else:
                current_load += weight
        
        return days_needed <= days
    
    left = max(weights)  # Must carry heaviest package
    right = sum(weights)  # Carry all at once
    
    while left < right:
        mid = left + (right - left) // 2
        
        if can_ship(mid):
            right = mid  # Try smaller capacity
        else:
            left = mid + 1  # Need larger capacity
    
    return left

# Time: O(n log(sum - max))
# Space: O(1)

# Example
weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
days = 5
print(ship_within_days(weights, days))  # 15
```

---

### 6.2.4 Binary Search Decision Tree

```
Is array sorted?
│
├─ YES → Can use binary search directly
│   ├─ Find exact value? → Standard template
│   ├─ Find first occurrence? → Leftmost template
│   ├─ Find last occurrence? → Rightmost template
│   └─ Find insertion point? → Leftmost template
│
├─ PARTIALLY (rotated, peak, etc.)
│   ├─ Rotated sorted? → Modified binary search
│   ├─ Mountain array? → Find peak then search
│   └─ Multiple sorted subarrays? → Determine sorted half
│
└─ NO, but answer space sorted
    ├─ "Minimum capacity"? → Binary search on answer
    ├─ "Maximum value that works"? → Binary search on answer
    └─ Optimization problem? → Binary search on answer
```

---

## Practice Questions - Section 6.1 (Sorting)

### Fill in the Gaps

1. Merge sort has ________ time complexity in all cases.
2. Quick sort's worst case occurs when the pivot is always the ________ element.
3. A ________ sorting algorithm preserves the relative order of equal elements.
4. Counting sort works in ________ time when the range k is not too large.
5. The ________ algorithm is optimal for nearly sorted arrays.

### True or False

1. Quick sort is always faster than merge sort. **[T/F]**
2. Heap sort guarantees O(n log n) and uses O(1) extra space. **[T/F]**
3. Radix sort is a comparison-based sorting algorithm. **[T/F]**
4. Insertion sort performs well on small or nearly sorted arrays. **[T/F]**
5. All O(n log n) sorting algorithms are stable. **[T/F]**

### Multiple Choice

1. Which sort is best for linked lists?
   - A) Quick sort
   - B) Merge sort
   - C) Heap sort
   - D) Selection sort

2. What's the space complexity of merge sort?
   - A) O(1)
   - B) O(log n)
   - C) O(n)
   - D) O(n log n)

3. Dutch National Flag algorithm is used in:
   - A) Merge sort
   - B) Quick sort (3-way partition)
   - C) Heap sort
   - D) Counting sort

### Code Challenge

```python
def merge_intervals(intervals):
    """
    LeetCode 56: Merge Intervals
    
    Merge overlapping intervals.
    
    Example: [[1,3],[2,6],[8,10],[15,18]]
    Output: [[1,6],[8,10],[15,18]]
    
    Hint: Sort first, then merge.
    """
    # Your code here
    pass
```

---

## Practice Questions - Section 6.2 (Binary Search)

### Fill in the Gaps

1. Binary search requires the array to be ________.
2. The time complexity of binary search is ________.
3. To avoid integer overflow, calculate mid as left + ________.
4. Binary search on answer is used when the answer space is ________.
5. In rotated sorted array, one half is always ________.

### True or False

1. Binary search can only find exact matches. **[T/F]**
2. Binary search works on linked lists efficiently. **[T/F]**
3. You can binary search on answer space even if array isn't sorted. **[T/F]**
4. The leftmost binary search finds the first occurrence. **[T/F]**
5. Binary search always uses O(1) space. **[T/F]**

### Multiple Choice

1. What's the maximum number of comparisons for binary search in array of size 16?
   - A) 4
   - B) 5
   - C) 8
   - D) 16

2. Which problem uses binary search on answer?
   - A) Find target in sorted array
   - B) Koko eating bananas
   - C) Two sum
   - D) Valid palindrome

3. In rotated sorted array [4,5,6,7,0,1,2], what's special?
   - A) No duplicates
   - B) One half always sorted
   - C) Pivot always at middle
   - D) Can't use binary search

### Code Challenge

```python
def find_peak_element(nums):
    """
    LeetCode 162: Find Peak Element
    
    Peak element is greater than its neighbors.
    Array may contain multiple peaks, return any.
    
    Example: [1,2,3,1] → 2 (index of peak 3)
    Example: [1,2,1,3,5,6,4] → 5 (index of peak 6)
    
    Solve in O(log n) using binary search.
    """
    # Your code here
    pass
```

---

## Answers - Section 6.1 (Sorting)

<details>
<summary><strong>View Answers</strong></summary>

### Fill in the Gaps

1. **O(n log n)**
2. **smallest** or **largest** (always at an end)
3. **stable**
4. **O(n + k)** or **linear**
5. **insertion sort**

### True or False

1. **False** - Quick sort has O(n²) worst case; merge sort always O(n log n)
2. **True** - Guaranteed time and in-place
3. **False** - Non-comparison based (uses digit positions)
4. **True** - O(n) for nearly sorted, good for small n
5. **False** - Quick sort and heap sort are not stable

### Multiple Choice

1. **B** - Merge sort works well without random access
2. **C** - O(n) for temporary arrays during merging
3. **B** - Quick sort variant for 3 distinct values

### Code Challenge Answer

```python
def merge_intervals(intervals):
    if not intervals:
        return []
    
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        
        if current[0] <= last[1]:
            # Overlapping: merge
            last[1] = max(last[1], current[1])
        else:
            # Non-overlapping: add new interval
            merged.append(current)
    
    return merged

# Time: O(n log n) for sorting
# Space: O(n) for result

# Example
intervals = [[1,3],[2,6],[8,10],[15,18]]
print(merge_intervals(intervals))
# [[1,6], [8,10], [15,18]]
```

</details>

---

## Answers - Section 6.2 (Binary Search)

<details>
<summary><strong>View Answers</strong></summary>

### Fill in the Gaps

1. **sorted**
2. **O(log n)**
3. **(right - left) // 2**
4. **sorted** or **monotonic**
5. **sorted**

### True or False

1. **False** - Can find insertion points, bounds, etc.
2. **False** - Needs random access; O(n) on linked lists
3. **True** - Answer space can be searched even if array isn't
4. **True** - Finds leftmost occurrence or insertion point
5. **True** - Iterative version uses O(1); recursive uses O(log n)

### Multiple Choice

1. **B** - log₂(16) = 4, but need 5 comparisons (0-4 levels)
2. **B** - Classic binary search on answer problem
3. **B** - Key insight for solving rotated array problems

### Code Challenge Answer

```python
def find_peak_element(nums):
    """
    Binary search: Move toward greater neighbor.
    Peak always exists at boundary or where nums[i] > nums[i+1].
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] > nums[mid + 1]:
            # Descending: peak is at mid or left
            right = mid
        else:
            # Ascending: peak is to the right
            left = mid + 1
    
    return left

# Time: O(log n)
# Space: O(1)

# Example
print(find_peak_element([1,2,3,1]))  # 2
print(find_peak_element([1,2,1,3,5,6,4]))  # 5
```

</details>

---

## LeetCode Problems - Sorting & Searching (NeetCode.io)

### Sorting - Easy ✅
- 88. Merge Sorted Array
- 242. Valid Anagram
- 349. Intersection of Two Arrays

### Sorting - Medium 🟨
- 56. Merge Intervals (IMPORTANT)
- 57. Insert Interval
- 75. Sort Colors (Dutch National Flag)
- 147. Insertion Sort List
- 148. Sort List
- 179. Largest Number
- 215. Kth Largest Element
- 274. H-Index
- 324. Wiggle Sort II
- 347. Top K Frequent Elements

### Sorting - Hard 🔴
- 315. Count of Smaller Numbers After Self
- 493. Reverse Pairs

### Binary Search - Easy ✅
- 35. Search Insert Position
- 69. Sqrt(x)
- 278. First Bad Version
- 704. Binary Search

### Binary Search - Medium 🟨
- 33. Search in Rotated Sorted Array (IMPORTANT)
- 34. Find First and Last Position (IMPORTANT)
- 74. Search a 2D Matrix
- 81. Search in Rotated Sorted Array II
- 153. Find Minimum in Rotated Sorted Array (IMPORTANT)
- 162. Find Peak Element
- 240. Search a 2D Matrix II
- 875. Koko Eating Bananas (IMPORTANT - Binary Search on Answer)
- 981. Time Based Key-Value Store
- 1011. Capacity To Ship Packages (Binary Search on Answer)

### Binary Search - Hard 🔴
- 4. Median of Two Sorted Arrays
- 410. Split Array Largest Sum

---

## Summary

### Sorting Quick Reference

**Choose sorting algorithm based on:**
- Array size: Small → Insertion, Large → Merge/Quick
- Memory: Limited → Heap/Quick, Plenty → Merge
- Stability needed: Merge, Insertion
- Data type: Integers in range → Counting/Radix
- Nearly sorted: Insertion, Bubble

### Binary Search Quick Reference

**Pattern recognition:**
- "Find target in sorted array" → Standard template
- "Find first/last occurrence" → Leftmost/Rightmost template
- "Minimum capacity/maximum value" → Binary search on answer
- "Rotated/Mountain array" → Modified binary search

**Key tips:**
- Always check if sorted (or answer space sorted)
- Use `left + (right - left) // 2` for mid
- Decide on `<=` vs `<` based on template
- Think about edge cases: empty array, single element

---

*Continue to: [7. Trees →](07-trees.md)*