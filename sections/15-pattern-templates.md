# DSA Pattern Templates - Quick Reference

> **Purpose**: Copy-paste ready templates for common problem-solving patterns. Use these as starting points when you recognize the pattern.

---

## Table of Contents

1. [Two Pointers Templates](#two-pointers-templates)
2. [Sliding Window Templates](#sliding-window-templates)
3. [Binary Search Templates](#binary-search-templates)
4. [Backtracking Templates](#backtracking-templates)
5. [Dynamic Programming Templates](#dynamic-programming-templates)
6. [Graph Traversal Templates](#graph-traversal-templates)

---

## Two Pointers Templates

### Template 1: Opposite Direction (Converging Pointers)

**Use When**: Sorted array, finding pairs/triplets, palindrome checking, partitioning

```python
def two_pointers_opposite_direction(arr, target):
    """
    Two pointers starting from opposite ends, moving toward each other.
    
    Common uses:
    - Two Sum (sorted array)
    - 3Sum, 4Sum
    - Container With Most Water
    - Trapping Rain Water
    - Valid Palindrome
    
    Time: O(n)
    Space: O(1)
    """
    left = 0
    right = len(arr) - 1
    result = []  # or other result type
    
    while left < right:
        # Calculate current state
        current = calculate(arr[left], arr[right])
        
        # Check condition
        if condition_met(current, target):
            result.append((arr[left], arr[right]))
            left += 1
            right -= 1
            
            # Optional: Skip duplicates
            while left < right and arr[left] == arr[left - 1]:
                left += 1
            while left < right and arr[right] == arr[right + 1]:
                right -= 1
                
        elif current < target:
            left += 1   # Need larger value
        else:
            right -= 1  # Need smaller value
    
    return result
```

**Specific: Two Sum (Sorted Array)**
```python
def two_sum_sorted(numbers, target):
    left, right = 0, len(numbers) - 1
    
    while left < right:
        current_sum = numbers[left] + numbers[right]
        if current_sum == target:
            return [left + 1, right + 1]  # 1-indexed
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []
```

**Specific: 3Sum**
```python
def three_sum(nums):
    nums.sort()
    result = []
    
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue  # Skip duplicates
        
        left, right = i + 1, len(nums) - 1
        target = -nums[i]
        
        while left < right:
            current = nums[left] + nums[right]
            if current == target:
                result.append([nums[i], nums[left], nums[right]])
                left += 1
                right -= 1
                while left < right and nums[left] == nums[left - 1]:
                    left += 1
                while left < right and nums[right] == nums[right + 1]:
                    right -= 1
            elif current < target:
                left += 1
            else:
                right -= 1
    
    return result
```

---

### Template 2: Same Direction (Slow & Fast Pointers)

**Use When**: In-place array modification, removing duplicates, partitioning

```python
def two_pointers_same_direction(arr):
    """
    Two pointers moving in same direction at different speeds.
    
    Common uses:
    - Remove Duplicates from Sorted Array
    - Move Zeroes
    - Remove Element
    - Partition Array
    
    Slow pointer: write position
    Fast pointer: read position
    
    Time: O(n)
    Space: O(1)
    """
    slow = 0  # Write pointer
    
    for fast in range(len(arr)):
        # Process element at fast pointer
        if should_keep(arr[fast]):
            arr[slow] = arr[fast]
            slow += 1
    
    return slow  # New length
```

**Specific: Remove Duplicates**
```python
def remove_duplicates(nums):
    if not nums:
        return 0
    
    slow = 1  # First element always unique
    
    for fast in range(1, len(nums)):
        if nums[fast] != nums[fast - 1]:
            nums[slow] = nums[fast]
            slow += 1
    
    return slow
```

**Specific: Move Zeroes**
```python
def move_zeroes(nums):
    slow = 0  # Position for next non-zero
    
    # Move all non-zeros to front
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow] = nums[fast]
            slow += 1
    
    # Fill rest with zeros
    for i in range(slow, len(nums)):
        nums[i] = 0
```

---

### Template 3: Fast & Slow (Floyd's Cycle Detection)

**Use When**: Linked list cycles, finding middle element, detecting patterns

```python
def fast_slow_pointers(head):
    """
    Floyd's Cycle Detection / Tortoise and Hare algorithm.
    
    Common uses:
    - Detect cycle in linked list
    - Find middle of linked list
    - Find duplicate number
    - Happy number problem
    
    Time: O(n)
    Space: O(1)
    """
    if not head or not head.next:
        return None
    
    slow = head
    fast = head
    
    # Phase 1: Detect if cycle exists
    while fast and fast.next:
        slow = slow.next         # Move 1 step
        fast = fast.next.next    # Move 2 steps
        
        if slow == fast:
            # Cycle detected
            return find_cycle_start(head, slow)
    
    return None  # No cycle

def find_cycle_start(head, meeting_point):
    """Find where cycle begins"""
    slow = head
    fast = meeting_point
    
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow  # Cycle start
```

**Specific: Find Middle of Linked List**
```python
def find_middle(head):
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow  # Middle node (or second middle if even length)
```

---

## Sliding Window Templates

### Template 1: Fixed-Size Window

**Use When**: Maximum/minimum of all K-size subarrays, average of K elements

```python
def fixed_window(arr, k):
    """
    Sliding window with fixed size k.
    
    Common uses:
    - Maximum sum of subarray of size K
    - Maximum average subarray
    - Number of subarrays of size K with condition
    
    Time: O(n)
    Space: O(1)
    """
    if len(arr) < k:
        return None
    
    # Step 1: Calculate first window
    window_sum = sum(arr[:k])
    result = window_sum  # or max, min, etc.
    
    # Step 2: Slide window
    for i in range(k, len(arr)):
        # Remove left element, add right element
        window_sum = window_sum - arr[i - k] + arr[i]
        # Update result
        result = max(result, window_sum)
    
    return result
```

**Specific: Maximum Sum Subarray of Size K**
```python
def max_sum_subarray(nums, k):
    window_sum = sum(nums[:k])
    max_sum = window_sum
    
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]
        max_sum = max(max_sum, window_sum)
    
    return max_sum
```

---

### Template 2: Variable-Size Window (Maximum)

**Use When**: Longest substring/subarray with condition

```python
def variable_window_maximum(arr):
    """
    Variable-size sliding window for LONGEST valid window.
    
    Common uses:
    - Longest substring without repeating characters
    - Longest substring with at most K distinct characters
    - Longest repeating character replacement
    - Max consecutive ones
    
    Time: O(n)
    Space: O(k) for window state
    """
    left = 0
    window_state = {}  # Hash map, set, or counter
    max_length = 0
    
    for right in range(len(arr)):
        # Expand window: add arr[right]
        add_to_window(arr[right], window_state)
        
        # Contract window if invalid
        while window_is_invalid(window_state):
            remove_from_window(arr[left], window_state)
            left += 1
        
        # Update maximum length
        max_length = max(max_length, right - left + 1)
    
    return max_length
```

**Specific: Longest Substring Without Repeating Characters**
```python
def length_of_longest_substring(s):
    char_index = {}
    max_length = 0
    left = 0
    
    for right in range(len(s)):
        if s[right] in char_index and char_index[s[right]] >= left:
            left = char_index[s[right]] + 1
        
        char_index[s[right]] = right
        max_length = max(max_length, right - left + 1)
    
    return max_length
```

---

### Template 3: Variable-Size Window (Minimum)

**Use When**: Shortest substring/subarray with condition

```python
def variable_window_minimum(arr, target):
    """
    Variable-size sliding window for SHORTEST valid window.
    
    Common uses:
    - Minimum window substring
    - Minimum size subarray sum
    - Smallest subarray with sum >= target
    
    Time: O(n)
    Space: O(k) for window state
    """
    left = 0
    window_state = {}
    min_length = float('inf')
    
    for right in range(len(arr)):
        # Expand window: add arr[right]
        add_to_window(arr[right], window_state)
        
        # Contract window while valid
        while window_is_valid(window_state, target):
            min_length = min(min_length, right - left + 1)
            remove_from_window(arr[left], window_state)
            left += 1
    
    return min_length if min_length != float('inf') else 0
```

**Specific: Minimum Window Substring**
```python
from collections import Counter

def min_window(s, t):
    if not s or not t:
        return ""
    
    required = Counter(t)
    needed = len(required)
    formed = 0
    
    window_counts = {}
    left = 0
    min_len = float('inf')
    result = (0, 0)
    
    for right in range(len(s)):
        char = s[right]
        window_counts[char] = window_counts.get(char, 0) + 1
        
        if char in required and window_counts[char] == required[char]:
            formed += 1
        
        while left <= right and formed == needed:
            if right - left + 1 < min_len:
                min_len = right - left + 1
                result = (left, right)
            
            char = s[left]
            window_counts[char] -= 1
            if char in required and window_counts[char] < required[char]:
                formed -= 1
            left += 1
    
    l, r = result
    return s[l:r+1] if min_len != float('inf') else ""
```

---

### Template 4: Sliding Window with Hash Map

**Use When**: K distinct elements, frequency constraints, anagram detection

```python
from collections import defaultdict

def sliding_window_hashmap(arr, k):
    """
    Sliding window with frequency counting.
    
    Common uses:
    - K distinct characters
    - Permutation in string
    - Find all anagrams
    - Frequency matching
    
    Time: O(n)
    Space: O(k)
    """
    window = defaultdict(int)
    left = 0
    result = 0
    
    for right in range(len(arr)):
        # Expand window
        window[arr[right]] += 1
        
        # Contract if constraint violated
        while len(window) > k:  # or other condition
            window[arr[left]] -= 1
            if window[arr[left]] == 0:
                del window[arr[left]]
            left += 1
        
        # Update result
        result = max(result, right - left + 1)
    
    return result
```

**Specific: Longest Substring with At Most K Distinct Characters**
```python
def length_of_longest_substring_k_distinct(s, k):
    from collections import defaultdict
    
    char_count = defaultdict(int)
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        char_count[s[right]] += 1
        
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        
        max_length = max(max_length, right - left + 1)
    
    return max_length
```

---

## Binary Search Templates

### Template 1: Standard Binary Search

```python
def binary_search(arr, target):
    """
    Standard binary search in sorted array.
    
    Returns: index if found, -1 if not found
    
    Time: O(log n)
    Space: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2  # Avoid overflow
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
```

### Template 2: Leftmost Binary Search

```python
def binary_search_leftmost(arr, target):
    """
    Find leftmost (first) occurrence of target.
    
    Returns: leftmost index, or -1 if not found
    """
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result
```

### Template 3: Binary Search on Answer

```python
def binary_search_on_answer(arr, condition):
    """
    Binary search on answer space.
    
    Use when: Finding minimum/maximum value that satisfies condition
    
    Common uses:
    - Koko eating bananas
    - Capacity to ship packages
    - Split array largest sum
    """
    left, right = min_possible_answer, max_possible_answer
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if is_valid(arr, mid):  # Can we achieve with mid?
            result = mid
            right = mid - 1  # Try smaller
        else:
            left = mid + 1  # Need larger
    
    return result
```

---

## Quick Pattern Recognition Guide

```
SORTED ARRAY + PAIR/TRIPLET → Two Pointers (opposite)
SUBARRAY/SUBSTRING → Sliding Window
LINKED LIST CYCLE → Fast & Slow Pointers
IN-PLACE MODIFICATION → Two Pointers (same direction)
SORTED ARRAY SEARCH → Binary Search
K-SIZE WINDOW → Fixed Sliding Window
LONGEST WITH CONDITION → Variable Sliding Window (max)
SHORTEST WITH CONDITION → Variable Sliding Window (min)
K DISTINCT ELEMENTS → Sliding Window with Hash Map
```

---

## Usage Tips

1. **Identify the pattern** from problem keywords
2. **Copy the template** that matches
3. **Modify the condition** for your specific problem
4. **Test with examples** to verify logic
5. **Handle edge cases** (empty input, size 1, etc.)

---

*Last Updated: January 2025*