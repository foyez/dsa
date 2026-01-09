# 5. Recursion & Backtracking

## 5.1 Recursion

### 5.1.1 Recursion Fundamentals

**Definition**: Recursion is a programming technique where a function calls itself to solve a problem by breaking it down into smaller, similar subproblems. Each recursive call works on a simpler version of the original problem until reaching a **base case** that can be solved directly without further recursion.

**Key Components of Recursion:**

1. **Base Case**: The terminating condition that stops recursion
   - Prevents infinite recursion
   - Returns a direct answer without recursive calls
   - Must be reached eventually

2. **Recursive Case**: The part where function calls itself
   - Works on a smaller/simpler subproblem
   - Moves toward the base case
   - Combines results from recursive calls

3. **Progress Toward Base Case**: Each call must move closer to termination
   - Smaller input size, fewer elements, simpler state
   - Guarantees eventual termination

**Structure**:
```python
def recursive_function(parameters):
    # Base case - stop recursion
    if base_condition:
        return base_value
    
    # Recursive case - call itself with simpler problem
    # Do some work
    result = recursive_function(modified_parameters)
    # Combine results
    return result
```

**Real-World Analogies**:
- **Russian nesting dolls**: Each doll contains a smaller doll until the smallest one
- **File system**: Folders contain folders until reaching files
- **Organizational hierarchy**: CEO → VP → Manager → Employee
- **Factorial**: n! = n × (n-1)! until reaching 1! = 1

**How Recursion Works - Call Stack**:
```
Recursive call: factorial(3)

Call Stack:
┌─────────────────┐
│ factorial(3)    │ ← Wait for factorial(2)
│ return 3 * ?    │
├─────────────────┤
│ factorial(2)    │ ← Wait for factorial(1)
│ return 2 * ?    │
├─────────────────┤
│ factorial(1)    │ ← Base case reached!
│ return 1        │ ✓ Start unwinding
└─────────────────┘

Unwinding:
factorial(1) returns 1
factorial(2) returns 2 * 1 = 2
factorial(3) returns 3 * 2 = 6
```

---

### 5.1.2 Basic Recursion Examples

#### Example 1: Factorial

```python
def factorial(n):
    """
    Calculate n! = n × (n-1) × ... × 2 × 1
    
    Base case: 0! = 1, 1! = 1
    Recursive case: n! = n × (n-1)!
    
    Example: 5! = 5 × 4! = 5 × 24 = 120
    """
    # Base case
    if n <= 1:
        return 1
    
    # Recursive case
    return n * factorial(n - 1)

# Time: O(n) - n recursive calls
# Space: O(n) - n stack frames

# Examples
print(factorial(5))  # 120
print(factorial(0))  # 1

# Trace for factorial(3):
# factorial(3) = 3 * factorial(2)
# factorial(2) = 2 * factorial(1)
# factorial(1) = 1 (base case)
# Returns: 1 → 2*1=2 → 3*2=6
```

#### Example 2: Fibonacci Numbers

```python
def fibonacci(n):
    """
    Calculate nth Fibonacci number.
    
    F(0) = 0, F(1) = 1
    F(n) = F(n-1) + F(n-2)
    
    Sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21...
    """
    # Base cases
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    # Recursive case
    return fibonacci(n - 1) + fibonacci(n - 2)

# Time: O(2^n) - exponential! Each call makes 2 calls
# Space: O(n) - maximum recursion depth

# Example
print(fibonacci(6))  # 8

# Call tree for fibonacci(4):
#           fib(4)
#         /        \
#     fib(3)      fib(2)
#     /    \      /    \
# fib(2) fib(1) fib(1) fib(0)
#  /  \
# fib(1) fib(0)
# Many redundant calculations! (fib(2) computed twice)
```

**Optimized with Memoization**:
```python
def fibonacci_memo(n, memo=None):
    """
    Fibonacci with memoization to avoid redundant calculations.
    
    Memoization: Store results of expensive function calls
    and return cached result when same inputs occur again.
    """
    if memo is None:
        memo = {}
    
    # Check cache first
    if n in memo:
        return memo[n]
    
    # Base cases
    if n <= 1:
        return n
    
    # Recursive case with memoization
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]

# Time: O(n) - each fib number calculated once
# Space: O(n) - memo dictionary + recursion stack

print(fibonacci_memo(50))  # Fast! 12586269025
```

#### Example 3: Power Function

```python
def power(base, exp):
    """
    Calculate base^exp using recursion.
    
    Base case: base^0 = 1
    Recursive: base^n = base × base^(n-1)
    """
    # Base case
    if exp == 0:
        return 1
    
    # Recursive case
    return base * power(base, exp - 1)

# Time: O(n) where n = exp
# Space: O(n) - recursion stack

print(power(2, 5))  # 32

# Optimized: Fast Exponentiation
def power_optimized(base, exp):
    """
    Fast exponentiation using divide and conquer.
    
    base^n = (base^(n/2))^2 if n is even
    base^n = base × (base^(n-1)) if n is odd
    """
    if exp == 0:
        return 1
    if exp == 1:
        return base
    
    # Divide and conquer
    half = power_optimized(base, exp // 2)
    
    if exp % 2 == 0:
        return half * half
    else:
        return base * half * half

# Time: O(log n) - divide exp by 2 each time
# Space: O(log n) - recursion depth

print(power_optimized(2, 10))  # 1024
```

---

### 5.1.3 Recursive Data Structure Traversal

#### Binary Tree Traversal

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorder_traversal(root):
    """
    Inorder: Left → Root → Right
    
    Produces sorted order for BST.
    """
    if not root:
        return []
    
    result = []
    result.extend(inorder_traversal(root.left))   # Left
    result.append(root.val)                        # Root
    result.extend(inorder_traversal(root.right))  # Right
    
    return result

def preorder_traversal(root):
    """
    Preorder: Root → Left → Right
    
    Used to create copy of tree.
    """
    if not root:
        return []
    
    result = []
    result.append(root.val)                        # Root
    result.extend(preorder_traversal(root.left))   # Left
    result.extend(preorder_traversal(root.right))  # Right
    
    return result

def postorder_traversal(root):
    """
    Postorder: Left → Right → Root
    
    Used to delete tree (delete children before parent).
    """
    if not root:
        return []
    
    result = []
    result.extend(postorder_traversal(root.left))   # Left
    result.extend(postorder_traversal(root.right))  # Right
    result.append(root.val)                         # Root
    
    return result

# Example tree:
#       1
#      / \
#     2   3
#    / \
#   4   5

root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)

print(inorder_traversal(root))    # [4, 2, 5, 1, 3]
print(preorder_traversal(root))   # [1, 2, 4, 5, 3]
print(postorder_traversal(root))  # [4, 5, 2, 3, 1]
```

#### Tree Height and Depth

```python
def max_depth(root):
    """
    LeetCode 104: Maximum Depth of Binary Tree
    
    Find height of tree (number of edges on longest path).
    
    Base case: Empty tree has depth 0
    Recursive: 1 + max(left_depth, right_depth)
    """
    if not root:
        return 0
    
    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)
    
    return 1 + max(left_depth, right_depth)

# Time: O(n) - visit each node once
# Space: O(h) - recursion depth = tree height

def is_balanced(root):
    """
    LeetCode 110: Balanced Binary Tree
    
    Check if tree is height-balanced (left and right subtrees
    of every node differ in height by at most 1).
    """
    def check_height(node):
        if not node:
            return 0
        
        left_height = check_height(node.left)
        if left_height == -1:
            return -1  # Left subtree not balanced
        
        right_height = check_height(node.right)
        if right_height == -1:
            return -1  # Right subtree not balanced
        
        # Check if current node is balanced
        if abs(left_height - right_height) > 1:
            return -1
        
        return 1 + max(left_height, right_height)
    
    return check_height(root) != -1

# Time: O(n)
# Space: O(h)
```

---

### 5.1.4 Recursion Patterns & Templates

#### Pattern 1: Single Branch Recursion

```python
def single_branch_template(n):
    """
    Template for problems with single recursive call.
    
    Use for: Factorial, sum of digits, reverse string
    """
    # Base case
    if base_condition(n):
        return base_value
    
    # Process current level
    current_result = process(n)
    
    # Single recursive call
    recursive_result = single_branch_template(next_state(n))
    
    # Combine results
    return combine(current_result, recursive_result)
```

**Example: Sum of Digits**
```python
def sum_of_digits(n):
    """
    Sum all digits of a number.
    
    Example: 1234 → 1 + 2 + 3 + 4 = 10
    """
    # Base case
    if n == 0:
        return 0
    
    # Current digit + sum of remaining digits
    return (n % 10) + sum_of_digits(n // 10)

# Time: O(log n) - number of digits
# Space: O(log n)

print(sum_of_digits(1234))  # 10
```

#### Pattern 2: Multiple Branch Recursion

```python
def multiple_branch_template(problem):
    """
    Template for problems with multiple recursive calls.
    
    Use for: Fibonacci, tree problems, combinations
    """
    # Base case
    if base_condition(problem):
        return base_value
    
    # Multiple recursive calls
    result1 = multiple_branch_template(subproblem1)
    result2 = multiple_branch_template(subproblem2)
    # ... more calls if needed
    
    # Combine results
    return combine(result1, result2)
```

**Example: Count Paths in Grid**
```python
def count_paths(m, n):
    """
    Count paths from top-left to bottom-right in m×n grid.
    Can only move right or down.
    
    LeetCode 62: Unique Paths
    """
    # Base case: 1×1 grid has 1 path
    if m == 1 or n == 1:
        return 1
    
    # Paths = paths from above + paths from left
    return count_paths(m - 1, n) + count_paths(m, n - 1)

# Time: O(2^(m+n)) - exponential without memoization
# Space: O(m + n) - recursion depth

# With memoization:
def count_paths_memo(m, n, memo=None):
    if memo is None:
        memo = {}
    
    if (m, n) in memo:
        return memo[(m, n)]
    
    if m == 1 or n == 1:
        return 1
    
    memo[(m, n)] = count_paths_memo(m-1, n, memo) + count_paths_memo(m, n-1, memo)
    return memo[(m, n)]

# Time: O(m × n)
# Space: O(m × n)
```

#### Pattern 3: Divide and Conquer

```python
def divide_and_conquer_template(arr, left, right):
    """
    Template for divide and conquer problems.
    
    Use for: Merge sort, quick sort, binary search
    """
    # Base case: single element or empty
    if left >= right:
        return base_value
    
    # Divide
    mid = left + (right - left) // 2
    
    # Conquer subproblems
    left_result = divide_and_conquer_template(arr, left, mid)
    right_result = divide_and_conquer_template(arr, mid + 1, right)
    
    # Combine results
    return merge(left_result, right_result)
```

**Example: Merge Sort**
```python
def merge_sort(arr):
    """
    Sort array using merge sort (divide and conquer).
    """
    # Base case
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]
    
    # Conquer (recursively sort)
    left_sorted = merge_sort(left)
    right_sorted = merge_sort(right)
    
    # Combine (merge sorted halves)
    return merge(left_sorted, right_sorted)

def merge(left, right):
    """Merge two sorted arrays"""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Time: O(n log n) - divide log n times, merge n elements each time
# Space: O(n) - temporary arrays during merge

arr = [38, 27, 43, 3, 9, 82, 10]
print(merge_sort(arr))  # [3, 9, 10, 27, 38, 43, 82]
```

---

### 5.1.5 Common Recursion Problems

#### Problem: Reverse String

```python
def reverse_string(s):
    """
    Reverse string using recursion.
    
    Base case: Empty or single character
    Recursive: Last char + reverse of rest
    """
    # Base case
    if len(s) <= 1:
        return s
    
    # Recursive case
    return s[-1] + reverse_string(s[:-1])

# Time: O(n²) - string concatenation is O(n)
# Space: O(n) - recursion stack

print(reverse_string("hello"))  # "olleh"

# Optimized version using list
def reverse_string_optimized(s, left=0, right=None):
    """In-place reversal using two pointers"""
    if right is None:
        right = len(s) - 1
    
    s = list(s)  # Convert to mutable list
    
    def helper(left, right):
        if left >= right:
            return
        s[left], s[right] = s[right], s[left]
        helper(left + 1, right - 1)
    
    helper(left, right)
    return ''.join(s)
```

#### Problem: Check Palindrome

```python
def is_palindrome_recursive(s, left=0, right=None):
    """
    Check if string is palindrome using recursion.
    
    Base case: Left >= right (checked all pairs)
    Recursive: First == last AND rest is palindrome
    """
    if right is None:
        right = len(s) - 1
    
    # Base case
    if left >= right:
        return True
    
    # Check current pair and recurse on middle
    if s[left] != s[right]:
        return False
    
    return is_palindrome_recursive(s, left + 1, right - 1)

# Time: O(n)
# Space: O(n) - recursion stack

print(is_palindrome_recursive("racecar"))  # True
print(is_palindrome_recursive("hello"))    # False
```

#### Problem: Generate Parentheses

```python
def generate_parentheses(n):
    """
    LeetCode 22: Generate Parentheses
    
    Generate all valid combinations of n pairs of parentheses.
    
    Example: n=3 → ["((()))","(()())","(())()","()(())","()()()"]
    
    Rules:
    - Can add '(' if open < n
    - Can add ')' if close < open
    """
    def backtrack(current, open_count, close_count):
        # Base case: valid combination complete
        if len(current) == 2 * n:
            result.append(current)
            return
        
        # Add '(' if we can
        if open_count < n:
            backtrack(current + '(', open_count + 1, close_count)
        
        # Add ')' if valid
        if close_count < open_count:
            backtrack(current + ')', open_count, close_count + 1)
    
    result = []
    backtrack('', 0, 0)
    return result

# Time: O(4^n / √n) - Catalan number
# Space: O(n) - recursion depth

print(generate_parentheses(3))
# ['((()))', '(()())', '(())()', '()(())', '()()()']
```

---

## 5.2 Backtracking

### 5.2.1 Backtracking Fundamentals

**Definition**: Backtracking is an algorithmic technique for solving problems by trying to build a solution incrementally, abandoning a solution ("backtracking") as soon as it determines that the solution cannot be completed. It's a refined brute force approach that systematically explores all possible solutions.

**Key Concepts:**

1. **Choice**: At each step, make a choice from available options
2. **Constraint**: Check if current choice is valid
3. **Goal**: Check if we've reached a solution
4. **Backtrack**: Undo choice and try next option

**Backtracking vs Recursion:**
- Recursion: General technique of function calling itself
- Backtracking: Specific use of recursion with systematic trial-and-error

**When to Use Backtracking:**
- Generate all combinations, permutations, or subsets
- Find all solutions satisfying constraints
- Puzzle solving (Sudoku, N-Queens)
- Path finding with obstacles
- Decision problems with multiple choices

**Real-World Analogies**:
- **Maze solving**: Try a path, if it's wrong, backtrack and try another
- **Puzzle solving**: Place a piece, if it doesn't fit later, remove and try different placement
- **Trial and error**: Try a solution, if it fails, undo and try alternative

**Backtracking Template**:
```python
def backtrack(state, choices, constraints):
    """
    Generic backtracking template.
    
    state: Current partial solution
    choices: Available options at current state
    constraints: Rules that must be satisfied
    """
    # Base case: reached goal
    if is_complete(state):
        result.append(copy(state))
        return
    
    # Try each choice
    for choice in get_choices(state):
        # Check constraint
        if is_valid(state, choice):
            # Make choice
            state.add(choice)
            
            # Recurse
            backtrack(state, choices, constraints)
            
            # Backtrack (undo choice)
            state.remove(choice)
```

**Visual Representation**:
```
Problem: Find all 2-number combinations from [1,2,3]

Decision Tree:
                    []
         /          |          \
       [1]         [2]         [3]
      /   \         |
   [1,2] [1,3]   [2,3]
   
Each path represents a decision.
When we reach length 2, we found a solution.
Backtrack to try other paths.
```

---

### 5.2.2 Backtracking Patterns

#### Pattern 1: Subsets (Choose or Not Choose)

```python
def subsets(nums):
    """
    LeetCode 78: Subsets
    
    Generate all possible subsets of unique elements.
    
    Example: [1,2,3] → [[], [1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]]
    
    Approach: For each element, choose to include or exclude it.
    """
    def backtrack(start, current):
        # Add current subset to result
        result.append(current[:])  # Copy current state
        
        # Try adding each remaining element
        for i in range(start, len(nums)):
            # Make choice: include nums[i]
            current.append(nums[i])
            
            # Recurse with next starting position
            backtrack(i + 1, current)
            
            # Backtrack: remove nums[i]
            current.pop()
    
    result = []
    backtrack(0, [])
    return result

# Time: O(2^n) - 2 choices for each of n elements
# Space: O(n) - recursion depth

print(subsets([1, 2, 3]))
# [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]
```

**With Duplicates**:
```python
def subsets_with_dup(nums):
    """
    LeetCode 90: Subsets II
    
    Generate subsets from array with duplicates.
    Avoid duplicate subsets in result.
    """
    def backtrack(start, current):
        result.append(current[:])
        
        for i in range(start, len(nums)):
            # Skip duplicates
            if i > start and nums[i] == nums[i-1]:
                continue
            
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()
    
    nums.sort()  # Sort to group duplicates
    result = []
    backtrack(0, [])
    return result

print(subsets_with_dup([1, 2, 2]))
# [[], [1], [1,2], [1,2,2], [2], [2,2]]
```

#### Pattern 2: Combinations (Choose K Elements)

```python
def combine(n, k):
    """
    LeetCode 77: Combinations
    
    Generate all combinations of k numbers from 1 to n.
    
    Example: n=4, k=2 → [[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]
    """
    def backtrack(start, current):
        # Base case: combination complete
        if len(current) == k:
            result.append(current[:])
            return
        
        # Pruning: not enough elements left
        # remaining = n - start + 1
        # needed = k - len(current)
        # if remaining < needed: return
        
        for i in range(start, n + 1):
            current.append(i)
            backtrack(i + 1, current)
            current.pop()
    
    result = []
    backtrack(1, [])
    return result

# Time: O(C(n,k) × k) - C(n,k) combinations, each takes k to copy
# Space: O(k) - recursion depth

print(combine(4, 2))
# [[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]
```

**Combination Sum**:
```python
def combination_sum(candidates, target):
    """
    LeetCode 39: Combination Sum
    
    Find all combinations that sum to target.
    Can reuse same number unlimited times.
    
    Example: candidates=[2,3,6,7], target=7 → [[2,2,3], [7]]
    """
    def backtrack(start, current, current_sum):
        # Base case: found target
        if current_sum == target:
            result.append(current[:])
            return
        
        # Pruning: exceeded target
        if current_sum > target:
            return
        
        for i in range(start, len(candidates)):
            current.append(candidates[i])
            # Can reuse same element, so pass i (not i+1)
            backtrack(i, current, current_sum + candidates[i])
            current.pop()
    
    result = []
    backtrack(0, [], 0)
    return result

# Time: O(2^target) in worst case
# Space: O(target) - recursion depth

print(combination_sum([2, 3, 6, 7], 7))
# [[2,2,3], [7]]
```

#### Pattern 3: Permutations (Arrange Elements)

```python
def permute(nums):
    """
    LeetCode 46: Permutations
    
    Generate all permutations of distinct numbers.
    
    Example: [1,2,3] → [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
    """
    def backtrack(current):
        # Base case: permutation complete
        if len(current) == len(nums):
            result.append(current[:])
            return
        
        for num in nums:
            # Skip if already used
            if num in current:
                continue
            
            current.append(num)
            backtrack(current)
            current.pop()
    
    result = []
    backtrack([])
    return result

# Time: O(n! × n) - n! permutations, each takes n to build
# Space: O(n) - recursion depth

# More efficient with used set:
def permute_optimized(nums):
    def backtrack(current, used):
        if len(current) == len(nums):
            result.append(current[:])
            return
        
        for i in range(len(nums)):
            if i in used:
                continue
            
            used.add(i)
            current.append(nums[i])
            backtrack(current, used)
            current.pop()
            used.remove(i)
    
    result = []
    backtrack([], set())
    return result

print(permute([1, 2, 3]))
# All 6 permutations
```

**Permutations with Duplicates**:
```python
def permute_unique(nums):
    """
    LeetCode 47: Permutations II
    
    Generate permutations with duplicates, no duplicate results.
    
    Example: [1,1,2] → [[1,1,2], [1,2,1], [2,1,1]]
    """
    def backtrack(current, used):
        if len(current) == len(nums):
            result.append(current[:])
            return
        
        for i in range(len(nums)):
            if used[i]:
                continue
            
            # Skip duplicate: if same as previous and previous not used
            if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                continue
            
            used[i] = True
            current.append(nums[i])
            backtrack(current, used)
            current.pop()
            used[i] = False
    
    nums.sort()  # Sort to group duplicates
    result = []
    used = [False] * len(nums)
    backtrack([], used)
    return result

print(permute_unique([1, 1, 2]))
# [[1,1,2], [1,2,1], [2,1,1]]
```

#### Pattern 4: Partition Problems

```python
def partition(s):
    """
    LeetCode 131: Palindrome Partitioning
    
    Partition string into palindrome substrings.
    
    Example: "aab" → [["a","a","b"], ["aa","b"]]
    """
    def is_palindrome(sub):
        return sub == sub[::-1]
    
    def backtrack(start, current):
        # Base case: partitioned entire string
        if start == len(s):
            result.append(current[:])
            return
        
        # Try all possible end positions
        for end in range(start + 1, len(s) + 1):
            substring = s[start:end]
            
            # Only continue if current substring is palindrome
            if is_palindrome(substring):
                current.append(substring)
                backtrack(end, current)
                current.pop()
    
    result = []
    backtrack(0, [])
    return result

# Time: O(n × 2^n) - 2^n partitions, each takes n to check palindrome
# Space: O(n)

print(partition("aab"))
# [['a', 'a', 'b'], ['aa', 'b']]
```

---

### 5.2.3 Classic Backtracking Problems

#### Problem: N-Queens

```python
def solve_n_queens(n):
    """
    LeetCode 51: N-Queens
    
    Place n queens on n×n board so no two attack each other.
    
    Queens attack same row, column, or diagonal.
    """
    def is_safe(row, col):
        # Check column
        for r in range(row):
            if board[r][col] == 'Q':
                return False
        
        # Check diagonal (top-left to bottom-right)
        r, c = row - 1, col - 1
        while r >= 0 and c >= 0:
            if board[r][c] == 'Q':
                return False
            r -= 1
            c -= 1
        
        # Check anti-diagonal (top-right to bottom-left)
        r, c = row - 1, col + 1
        while r >= 0 and c < n:
            if board[r][c] == 'Q':
                return False
            r -= 1
            c += 1
        
        return True
    
    def backtrack(row):
        # Base case: placed all queens
        if row == n:
            result.append([''.join(row) for row in board])
            return
        
        # Try placing queen in each column of current row
        for col in range(n):
            if is_safe(row, col):
                board[row][col] = 'Q'
                backtrack(row + 1)
                board[row][col] = '.'  # Backtrack
    
    board = [['.' for _ in range(n)] for _ in range(n)]
    result = []
    backtrack(0)
    return result

# Time: O(n!) - try all row permutations
# Space: O(n²) - board storage

# Example for n=4:
solutions = solve_n_queens(4)
for solution in solutions:
    for row in solution:
        print(row)
    print()
# Outputs two valid 4-queens solutions
```

#### Problem: Sudoku Solver

```python
def solve_sudoku(board):
    """
    LeetCode 37: Sudoku Solver
    
    Solve 9×9 Sudoku puzzle.
    """
    def is_valid(row, col, num):
        # Check row
        if num in board[row]:
            return False
        
        # Check column
        if num in [board[r][col] for r in range(9)]:
            return False
        
        # Check 3×3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if board[r][c] == num:
                    return False
        
        return True
    
    def backtrack():
        # Find next empty cell
        for row in range(9):
            for col in range(9):
                if board[row][col] == '.':
                    # Try each number
                    for num in '123456789':
                        if is_valid(row, col, num):
                            board[row][col] = num
                            
                            if backtrack():
                                return True
                            
                            board[row][col] = '.'  # Backtrack
                    
                    return False  # No valid number found
        
        return True  # All cells filled
    
    backtrack()

# Time: O(9^(n×n)) in worst case
# Space: O(1) - modify in place
```

#### Problem: Word Search

```python
def exist(board, word):
    """
    LeetCode 79: Word Search
    
    Find if word exists in grid by connecting adjacent cells.
    
    Example:
    board = [['A','B','C','E'],
             ['S','F','C','S'],
             ['A','D','E','E']]
    word = "ABCCED" → True
    """
    def backtrack(row, col, index):
        # Base case: found entire word
        if index == len(word):
            return True
        
        # Boundary check
        if (row < 0 or row >= len(board) or 
            col < 0 or col >= len(board[0]) or 
            board[row][col] != word[index]):
            return False
        
        # Mark as visited
        temp = board[row][col]
        board[row][col] = '#'
        
        # Try all 4 directions
        found = (backtrack(row + 1, col, index + 1) or
                backtrack(row - 1, col, index + 1) or
                backtrack(row, col + 1, index + 1) or
                backtrack(row, col - 1, index + 1))
        
        # Backtrack: restore cell
        board[row][col] = temp
        
        return found
    
    # Try starting from each cell
    for row in range(len(board)):
        for col in range(len(board[0])):
            if backtrack(row, col, 0):
                return True
    
    return False

# Time: O(m × n × 4^L) where L = word length
# Space: O(L) - recursion depth
```

---

## Recursion vs Iteration

| Aspect | Recursion | Iteration |
|--------|-----------|-----------|
| **Readability** | Often cleaner for tree/graph problems | Better for simple loops |
| **Memory** | O(n) stack space | O(1) typically |
| **Performance** | Function call overhead | Generally faster |
| **When to use** | Tree traversal, divide & conquer | Simple counting, iteration |
| **Termination** | Base case | Loop condition |

**Converting Recursion to Iteration**:
```python
# Recursive factorial
def factorial_recursive(n):
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)

# Iterative factorial
def factorial_iterative(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# Both: Time O(n), but iterative uses O(1) space
```

---

## Practice Questions - Section 5.1 (Recursion)

### Fill in the Gaps

1. The two essential components of recursion are the ________ case and the recursive case.
2. Each recursive call must make progress toward the ________ case.
3. The time complexity of naive Fibonacci recursion is ________.
4. Memoization improves Fibonacci to ________ time complexity.
5. The space complexity of recursion is determined by the maximum ________ depth.

### True or False

1. Every recursive function can be converted to an iterative one. **[T/F]**
2. Recursion always uses more memory than iteration. **[T/F]**
3. The base case prevents infinite recursion. **[T/F]**
4. Fast exponentiation reduces time complexity to O(log n). **[T/F]**
5. Inorder traversal of a BST produces sorted order. **[T/F]**

### Multiple Choice

1. What is the space complexity of recursive factorial?
   - A) O(1)
   - B) O(log n)
   - C) O(n)
   - D) O(n²)

2. Which traversal visits nodes in Root → Left → Right order?
   - A) Inorder
   - B) Preorder
   - C) Postorder
   - D) Level order

3. Divide and conquer is best for:
   - A) Linear search
   - B) Merge sort
   - C) Bubble sort
   - D) Sequential processing

### Code Challenge

```python
def climb_stairs(n):
    """
    LeetCode 70: Climbing Stairs
    
    You're climbing stairs. Takes n steps to reach top.
    Each time you can climb 1 or 2 steps.
    How many distinct ways can you climb to the top?
    
    Example: n=3 → 3 ways (1+1+1, 1+2, 2+1)
    
    Implement using recursion with memoization.
    """
    # Your code here
    pass
```

---

## Practice Questions - Section 5.2 (Backtracking)

### Fill in the Gaps

1. Backtracking systematically explores all ________ solutions.
2. The three steps in backtracking are: make choice, ________, and backtrack.
3. Generating all subsets of n elements takes ________ time.
4. The N-Queens problem uses backtracking to place queens without ________.
5. In backtracking, we ________ a choice if it violates constraints.

### True or False

1. Backtracking is more efficient than brute force. **[T/F]**
2. All backtracking problems have exponential time complexity. **[T/F]**
3. Permutations of n elements generate n! results. **[T/F]**
4. Backtracking can solve any constraint satisfaction problem. **[T/F]**
5. Pruning in backtracking reduces the search space. **[T/F]**

### Multiple Choice

1. How many subsets exist for an array of n elements?
   - A) n
   - B) n²
   - C) 2^n
   - D) n!

2. Which pattern is used for "Combination Sum"?
   - A) Subsets
   - B) Permutations
   - C) Combinations with reuse
   - D) Partitioning

3. The N-Queens problem for n=8 is:
   - A) Polynomial time
   - B) Linear time
   - C) NP-complete
   - D) Unsolvable

### Code Challenge

```python
def letter_combinations(digits):
    """
    LeetCode 17: Letter Combinations of Phone Number
    
    Given digit string, return all possible letter combinations.
    
    Mapping: 2→abc, 3→def, 4→ghi, 5→jkl, 
             6→mno, 7→pqrs, 8→tuv, 9→wxyz
    
    Example: digits = "23"
    Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
    
    Implement using backtracking.
    """
    # Your code here
    pass
```

---

## Answers - Section 5.1 (Recursion)

<details>
<summary><strong>View Answers</strong></summary>

### Fill in the Gaps

1. **base**
2. **base** (or terminating condition)
3. **O(2^n)** (exponential)
4. **O(n)**
5. **recursion** (or call stack)

### True or False

1. **True** - Any recursion can be simulated with a stack
2. **False** - Tail recursion can be optimized to O(1) space
3. **True** - Essential for termination
4. **True** - Reduces from O(n) to O(log n)
5. **True** - Left → Root → Right gives sorted order

### Multiple Choice

1. **C** - O(n) recursion depth for n calls
2. **B** - Preorder: Root → Left → Right
3. **B** - Merge sort uses divide and conquer

### Code Challenge Answer

```python
def climb_stairs(n, memo=None):
    """
    Base cases: 0 steps = 0 ways, 1 step = 1 way, 2 steps = 2 ways
    Recursive: ways(n) = ways(n-1) + ways(n-2)
    
    Like Fibonacci!
    """
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n <= 2:
        return n
    
    memo[n] = climb_stairs(n-1, memo) + climb_stairs(n-2, memo)
    return memo[n]

# Time: O(n)
# Space: O(n)

# Examples
print(climb_stairs(3))  # 3
print(climb_stairs(5))  # 8
```

</details>

---

## Answers - Section 5.2 (Backtracking)

<details>
<summary><strong>View Answers</strong></summary>

### Fill in the Gaps

1. **possible** (or candidate)
2. **recurse** (or explore)
3. **O(2^n)** (exponential)
4. **attacks** (or conflicts)
5. **prune** (or skip/abandon)

### True or False

1. **True** - Prunes invalid paths early
2. **False** - Some have polynomial solutions with pruning
3. **True** - n! permutations for n distinct elements
4. **True** - Though may be very slow
5. **True** - Eliminates impossible branches

### Multiple Choice

1. **C** - Each element: include or exclude = 2^n
2. **C** - Combinations where elements can be reused
3. **C** - Classic NP-complete problem

### Code Challenge Answer

```python
def letter_combinations(digits):
    if not digits:
        return []
    
    mapping = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }
    
    def backtrack(index, current):
        # Base case: combination complete
        if index == len(digits):
            result.append(current)
            return
        
        # Get letters for current digit
        letters = mapping[digits[index]]
        
        # Try each letter
        for letter in letters:
            backtrack(index + 1, current + letter)
    
    result = []
    backtrack(0, '')
    return result

# Time: O(4^n) - worst case ('7' and '9' have 4 letters)
# Space: O(n) - recursion depth

# Example
print(letter_combinations("23"))
# ['ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf']
```

</details>

---

## LeetCode Problems - Recursion & Backtracking (NeetCode.io)

### Recursion - Easy ✅
- 206. Reverse Linked List
- 231. Power of Two
- 326. Power of Three
- 509. Fibonacci Number

### Recursion - Medium 🟨
- 50. Pow(x, n) - Fast exponentiation
- 70. Climbing Stairs - Fibonacci variant
- 894. All Possible Full Binary Trees

### Backtracking - Medium 🟨
- 17. Letter Combinations of a Phone Number (IMPORTANT)
- 22. Generate Parentheses (IMPORTANT)
- 39. Combination Sum (IMPORTANT)
- 40. Combination Sum II
- 46. Permutations (IMPORTANT)
- 47. Permutations II
- 77. Combinations
- 78. Subsets (IMPORTANT)
- 90. Subsets II
- 131. Palindrome Partitioning
- 216. Combination Sum III

### Backtracking - Hard 🔴
- 37. Sudoku Solver
- 51. N-Queens (VERY IMPORTANT)
- 52. N-Queens II
- 79. Word Search (IMPORTANT)
- 212. Word Search II

---

## Summary: When to Use What

| Problem Type | Technique | Example |
|--------------|-----------|---------|
| **Tree traversal** | Recursion | Inorder, Preorder, Postorder |
| **Divide and conquer** | Recursion | Merge sort, Quick sort |
| **Generate all** | Backtracking | Subsets, Permutations |
| **Constraint satisfaction** | Backtracking | N-Queens, Sudoku |
| **Path finding** | Backtracking | Word Search, Maze |
| **Optimization** | Dynamic Programming | Later chapter |

---

*Continue to: [6. Sorting & Searching →](06-sorting-searching.md)*