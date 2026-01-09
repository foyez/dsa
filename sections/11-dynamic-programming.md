# 11. Dynamic Programming

## 11.1 Dynamic Programming Fundamentals

### 11.1.1 What is Dynamic Programming?

**Definition**: Dynamic Programming (DP) is an algorithmic technique for solving optimization problems by breaking them down into simpler subproblems and storing the results to avoid redundant computations. It combines the correctness of complete search with the efficiency of greedy algorithms.

**Key Characteristics:**

1. **Optimal Substructure**: Solution to problem can be constructed from solutions to subproblems
2. **Overlapping Subproblems**: Same subproblems solved multiple times
3. **Memoization/Tabulation**: Store subproblem results to avoid recomputation

**When to Use DP:**
- Problem asks for optimum (maximum/minimum)
- Problem asks for count of ways
- Problem asks for "is it possible"
- Brute force has overlapping subproblems

**DP vs Recursion vs Greedy:**
```
Recursion alone:     Recalculates subproblems (exponential)
DP (Memoization):    Stores results (polynomial)
Greedy:              Makes local optimal choice (may not be global optimal)
```

**Real-World Examples**:
- **Route planning**: Shortest path with multiple options
- **Resource allocation**: Knapsack, scheduling
- **String editing**: Autocorrect, diff tools
- **Game theory**: Chess, tic-tac-toe optimal play

---

### 11.1.2 Two Approaches: Memoization vs Tabulation

**Memoization (Top-Down)**:
- Start with original problem
- Recursively solve subproblems
- Store results in cache (usually dict/array)
- Also called "lazy evaluation"

```python
def fib_memo(n, memo=None):
    """
    Fibonacci with memoization (top-down).
    """
    if memo is None:
        memo = {}
    
    # Base cases
    if n <= 1:
        return n
    
    # Check cache
    if n in memo:
        return memo[n]
    
    # Compute and store
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# Time: O(n)
# Space: O(n) - memo + recursion stack
```

**Tabulation (Bottom-Up)**:
- Start with base cases
- Iteratively build up to solution
- Store results in table (usually array)
- Also called "eager evaluation"

```python
def fib_tab(n):
    """
    Fibonacci with tabulation (bottom-up).
    """
    if n <= 1:
        return n
    
    # Build table
    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

# Time: O(n)
# Space: O(n) - dp array only
```

**Space Optimization**:
```python
def fib_optimized(n):
    """
    Fibonacci with O(1) space.
    Only need last two values.
    """
    if n <= 1:
        return n
    
    prev2 = 0
    prev1 = 1
    
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1

# Time: O(n)
# Space: O(1)
```

**Comparison:**

| Aspect | Memoization | Tabulation |
|--------|-------------|------------|
| **Direction** | Top-down | Bottom-up |
| **Implementation** | Recursive | Iterative |
| **Space** | O(n) + stack | O(n) |
| **When to use** | Complex transitions | Simple iteration |
| **Advantages** | Only computes needed | No recursion overhead |

---

### 11.1.3 DP Problem-Solving Framework

**Step-by-Step Approach:**

1. **Identify if it's DP**
   - Optimization problem (max/min/count)
   - Overlapping subproblems
   - Optimal substructure

2. **Define the State**
   - What information do we need to track?
   - `dp[i]` = answer for subproblem ending at i
   - `dp[i][j]` = answer for subproblem with parameters i, j

3. **Write Recurrence Relation**
   - How to compute `dp[i]` from previous states?
   - Example: `dp[i] = dp[i-1] + dp[i-2]` (Fibonacci)

4. **Identify Base Cases**
   - Smallest subproblems we can solve directly
   - Example: `dp[0] = 0, dp[1] = 1` (Fibonacci)

5. **Determine Computation Order**
   - Which subproblems to solve first?
   - Usually: smaller indices before larger

6. **Optimize Space** (if needed)
   - Can we use O(1) space instead of O(n)?
   - Only keep necessary previous states

**Example: Climbing Stairs**
```python
def climb_stairs(n):
    """
    LeetCode 70: Climbing Stairs
    
    You're climbing stairs with n steps.
    Can climb 1 or 2 steps at a time.
    How many distinct ways to reach top?
    
    Step 1: Identify DP
    - Count number of ways (optimization)
    - ways(n) depends on ways(n-1) and ways(n-2)
    
    Step 2: Define State
    - dp[i] = number of ways to reach step i
    
    Step 3: Recurrence Relation
    - dp[i] = dp[i-1] + dp[i-2]
    - (Can reach step i from step i-1 or i-2)
    
    Step 4: Base Cases
    - dp[0] = 1 (one way to stay at ground)
    - dp[1] = 1 (one way to reach first step)
    
    Step 5: Order
    - Compute dp[0], dp[1], ..., dp[n] in order
    """
    if n <= 1:
        return 1
    
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

# Time: O(n)
# Space: O(n)

# Space optimized:
def climb_stairs_optimized(n):
    if n <= 1:
        return 1
    
    prev2 = 1
    prev1 = 1
    
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1
```

---

## 11.2 DP Pattern 1: 1D Array

### 11.2.1 House Robber

```python
def rob(nums):
    """
    LeetCode 198: House Robber
    
    Rob houses to maximize money. Can't rob adjacent houses.
    
    State: dp[i] = max money robbing houses 0..i
    Recurrence: dp[i] = max(dp[i-1], dp[i-2] + nums[i])
                        (skip house i OR rob house i)
    Base: dp[0] = nums[0], dp[1] = max(nums[0], nums[1])
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    
    for i in range(2, n):
        dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    
    return dp[n-1]

# Time: O(n)
# Space: O(n)

# Space optimized:
def rob_optimized(nums):
    if not nums:
        return 0
    
    prev2 = 0
    prev1 = 0
    
    for num in nums:
        current = max(prev1, prev2 + num)
        prev2 = prev1
        prev1 = current
    
    return prev1

# Example
print(rob([2, 7, 9, 3, 1]))  # 12 (rob 2, 9, 1)
```

---

### 11.2.2 Coin Change

```python
def coin_change(coins, amount):
    """
    LeetCode 322: Coin Change
    
    Find minimum coins to make amount.
    
    State: dp[i] = minimum coins to make amount i
    Recurrence: dp[i] = min(dp[i-coin] + 1) for all coins
    Base: dp[0] = 0 (0 coins for amount 0)
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if i >= coin:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

# Time: O(amount × coins)
# Space: O(amount)

# Example
print(coin_change([1, 2, 5], 11))  # 3 (5+5+1)
print(coin_change([2], 3))  # -1 (impossible)
```

---

### 11.2.3 Longest Increasing Subsequence

```python
def length_of_lis(nums):
    """
    LeetCode 300: Longest Increasing Subsequence
    
    Find length of longest strictly increasing subsequence.
    
    State: dp[i] = length of LIS ending at index i
    Recurrence: dp[i] = max(dp[j] + 1) for all j < i where nums[j] < nums[i]
    Base: dp[i] = 1 (each element is LIS of length 1)
    """
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

# Time: O(n²)
# Space: O(n)

# Optimized with binary search: O(n log n)
def length_of_lis_optimized(nums):
    """
    Use binary search to find insertion position.
    """
    import bisect
    
    tails = []  # tails[i] = smallest tail of LIS of length i+1
    
    for num in nums:
        pos = bisect.bisect_left(tails, num)
        
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    
    return len(tails)

# Example
print(length_of_lis([10, 9, 2, 5, 3, 7, 101, 18]))  # 4 ([2,3,7,101])
```

---

### 11.2.4 Word Break

```python
def word_break(s, word_dict):
    """
    LeetCode 139: Word Break
    
    Check if string can be segmented into words from dictionary.
    
    State: dp[i] = True if s[0:i] can be segmented
    Recurrence: dp[i] = True if dp[j] and s[j:i] in wordDict
    Base: dp[0] = True (empty string)
    """
    word_set = set(word_dict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    
    return dp[n]

# Time: O(n² × m) where m = average word length
# Space: O(n)

# Example
s = "leetcode"
word_dict = ["leet", "code"]
print(word_break(s, word_dict))  # True
```

---

## 11.3 DP Pattern 2: 2D Grid

### 11.3.1 Unique Paths

```python
def unique_paths(m, n):
    """
    LeetCode 62: Unique Paths
    
    Count paths from top-left to bottom-right in m×n grid.
    Can only move right or down.
    
    State: dp[i][j] = number of paths to reach cell (i,j)
    Recurrence: dp[i][j] = dp[i-1][j] + dp[i][j-1]
    Base: dp[0][j] = 1, dp[i][0] = 1 (one way along edges)
    """
    dp = [[0] * n for _ in range(m)]
    
    # Base cases
    for i in range(m):
        dp[i][0] = 1
    for j in range(n):
        dp[0][j] = 1
    
    # Fill table
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    
    return dp[m-1][n-1]

# Time: O(m × n)
# Space: O(m × n)

# Space optimized to O(n):
def unique_paths_optimized(m, n):
    dp = [1] * n
    
    for i in range(1, m):
        for j in range(1, n):
            dp[j] = dp[j] + dp[j-1]
    
    return dp[n-1]

# Example
print(unique_paths(3, 7))  # 28
```

---

### 11.3.2 Minimum Path Sum

```python
def min_path_sum(grid):
    """
    LeetCode 64: Minimum Path Sum
    
    Find path from top-left to bottom-right with minimum sum.
    Can only move right or down.
    
    State: dp[i][j] = minimum sum to reach (i,j)
    Recurrence: dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
    """
    if not grid:
        return 0
    
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    
    # Base case
    dp[0][0] = grid[0][0]
    
    # First row
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]
    
    # First column
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    
    # Fill rest
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
    
    return dp[m-1][n-1]

# Time: O(m × n)
# Space: O(m × n)

# Example
grid = [
    [1, 3, 1],
    [1, 5, 1],
    [4, 2, 1]
]
print(min_path_sum(grid))  # 7 (1→3→1→1→1)
```

---

### 11.3.3 Longest Common Subsequence

```python
def longest_common_subsequence(text1, text2):
    """
    LeetCode 1143: Longest Common Subsequence
    
    Find length of longest subsequence common to both strings.
    
    State: dp[i][j] = LCS length for text1[0:i] and text2[0:j]
    Recurrence:
      if text1[i-1] == text2[j-1]:
        dp[i][j] = dp[i-1][j-1] + 1
      else:
        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    Base: dp[0][j] = 0, dp[i][0] = 0
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

# Time: O(m × n)
# Space: O(m × n)

# Example
print(longest_common_subsequence("abcde", "ace"))  # 3 ("ace")
```

---

### 11.3.4 Edit Distance

```python
def min_distance(word1, word2):
    """
    LeetCode 72: Edit Distance (Levenshtein Distance)
    
    Minimum operations to convert word1 to word2.
    Operations: insert, delete, replace
    
    State: dp[i][j] = min operations for word1[0:i] → word2[0:j]
    Recurrence:
      if word1[i-1] == word2[j-1]:
        dp[i][j] = dp[i-1][j-1]
      else:
        dp[i][j] = 1 + min(
          dp[i-1][j],    # delete
          dp[i][j-1],    # insert
          dp[i-1][j-1]   # replace
        )
    Base: dp[i][0] = i, dp[0][j] = j
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # delete
                    dp[i][j-1],      # insert
                    dp[i-1][j-1]     # replace
                )
    
    return dp[m][n]

# Time: O(m × n)
# Space: O(m × n)

# Example
print(min_distance("horse", "ros"))  # 3
# horse → rorse (replace 'h' with 'r')
# rorse → rose (remove 'r')
# rose → ros (remove 'e')
```

---

## 11.4 DP Pattern 3: Knapsack

### 11.4.1 0/1 Knapsack

```python
def knapsack_01(weights, values, capacity):
    """
    0/1 Knapsack: Each item can be taken 0 or 1 time.
    
    State: dp[i][w] = max value using items 0..i-1 with capacity w
    Recurrence:
      if weights[i-1] <= w:
        dp[i][w] = max(
          dp[i-1][w],                           # don't take item i
          dp[i-1][w-weights[i-1]] + values[i-1] # take item i
        )
      else:
        dp[i][w] = dp[i-1][w]
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(
                    dp[i-1][w],
                    dp[i-1][w - weights[i-1]] + values[i-1]
                )
            else:
                dp[i][w] = dp[i-1][w]
    
    return dp[n][capacity]

# Time: O(n × capacity)
# Space: O(n × capacity)

# Space optimized to O(capacity):
def knapsack_01_optimized(weights, values, capacity):
    dp = [0] * (capacity + 1)
    
    for i in range(len(weights)):
        # Iterate backwards to avoid using updated values
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]

# Example
weights = [1, 3, 4, 5]
values = [1, 4, 5, 7]
capacity = 7
print(knapsack_01(weights, values, capacity))  # 9 (items 1 and 2)
```

---

### 11.4.2 Partition Equal Subset Sum

```python
def can_partition(nums):
    """
    LeetCode 416: Partition Equal Subset Sum
    
    Check if array can be partitioned into two subsets with equal sum.
    
    Approach: 0/1 knapsack variant
    - If total sum is odd, can't partition
    - Otherwise, find subset with sum = total/2
    
    State: dp[i] = True if sum i is achievable
    """
    total = sum(nums)
    
    if total % 2 != 0:
        return False
    
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        # Iterate backwards
        for i in range(target, num - 1, -1):
            dp[i] = dp[i] or dp[i - num]
    
    return dp[target]

# Time: O(n × sum)
# Space: O(sum)

# Example
print(can_partition([1, 5, 11, 5]))  # True ([1,5,5] and [11])
print(can_partition([1, 2, 3, 5]))   # False
```

---

### 11.4.3 Target Sum

```python
def find_target_sum_ways(nums, target):
    """
    LeetCode 494: Target Sum
    
    Assign + or - to each number to make sum equal target.
    Count number of ways.
    
    Approach: Transform to subset sum problem
    - Let P = positive subset, N = negative subset
    - P - N = target
    - P + N = sum(nums)
    - Therefore: P = (target + sum) / 2
    - Find number of subsets with sum P
    
    State: dp[i] = number of ways to make sum i
    """
    total = sum(nums)
    
    # Check if valid
    if abs(target) > total or (target + total) % 2 != 0:
        return 0
    
    subset_sum = (target + total) // 2
    
    dp = [0] * (subset_sum + 1)
    dp[0] = 1
    
    for num in nums:
        for i in range(subset_sum, num - 1, -1):
            dp[i] += dp[i - num]
    
    return dp[subset_sum]

# Time: O(n × sum)
# Space: O(sum)

# Example
print(find_target_sum_ways([1, 1, 1, 1, 1], 3))  # 5
# +1+1+1+1-1 = 3
# +1+1+1-1+1 = 3
# +1+1-1+1+1 = 3
# +1-1+1+1+1 = 3
# -1+1+1+1+1 = 3
```

---

## 11.5 DP Pattern 4: Subsequence Problems

### 11.5.1 Longest Palindromic Substring

```python
def longest_palindrome(s):
    """
    LeetCode 5: Longest Palindromic Substring
    
    Find longest palindromic substring.
    
    State: dp[i][j] = True if s[i:j+1] is palindrome
    Recurrence:
      dp[i][j] = (s[i] == s[j]) and (j-i < 2 or dp[i+1][j-1])
    """
    if not s:
        return ""
    
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    start = 0
    max_len = 1
    
    # Every single character is palindrome
    for i in range(n):
        dp[i][i] = True
    
    # Check substrings of length 2
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start = i
            max_len = 2
    
    # Check substrings of length 3+
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                start = i
                max_len = length
    
    return s[start:start + max_len]

# Time: O(n²)
# Space: O(n²)

# Optimized expand around center: O(n²) time, O(1) space
def longest_palindrome_expand(s):
    """
    Expand around each possible center.
    """
    def expand(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1
    
    start = 0
    max_len = 0
    
    for i in range(len(s)):
        # Odd length palindromes
        len1 = expand(i, i)
        # Even length palindromes
        len2 = expand(i, i + 1)
        
        length = max(len1, len2)
        if length > max_len:
            max_len = length
            start = i - (length - 1) // 2
    
    return s[start:start + max_len]

# Example
print(longest_palindrome("babad"))  # "bab" or "aba"
```

---

### 11.5.2 Palindromic Substrings

```python
def count_substrings(s):
    """
    LeetCode 647: Palindromic Substrings
    
    Count all palindromic substrings.
    
    State: dp[i][j] = True if s[i:j+1] is palindrome
    """
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    count = 0
    
    # Single characters
    for i in range(n):
        dp[i][i] = True
        count += 1
    
    # Length 2
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            count += 1
    
    # Length 3+
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                count += 1
    
    return count

# Time: O(n²)
# Space: O(n²)

# Example
print(count_substrings("abc"))  # 3 ("a", "b", "c")
print(count_substrings("aaa"))  # 6 ("a", "a", "a", "aa", "aa", "aaa")
```

---

### 11.5.3 Decode Ways

```python
def num_decodings(s):
    """
    LeetCode 91: Decode Ways
    
    '1' → 'A', '2' → 'B', ..., '26' → 'Z'
    Count ways to decode string.
    
    State: dp[i] = number of ways to decode s[0:i]
    Recurrence:
      - If s[i-1] != '0': dp[i] += dp[i-1] (single digit)
      - If 10 <= int(s[i-2:i]) <= 26: dp[i] += dp[i-2] (two digits)
    Base: dp[0] = 1 (empty string)
    """
    if not s or s[0] == '0':
        return 0
    
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1
    
    for i in range(2, n + 1):
        # Single digit
        if s[i-1] != '0':
            dp[i] += dp[i-1]
        
        # Two digits
        two_digit = int(s[i-2:i])
        if 10 <= two_digit <= 26:
            dp[i] += dp[i-2]
    
    return dp[n]

# Time: O(n)
# Space: O(n)

# Space optimized:
def num_decodings_optimized(s):
    if not s or s[0] == '0':
        return 0
    
    prev2 = 1
    prev1 = 1
    
    for i in range(2, len(s) + 1):
        current = 0
        
        if s[i-1] != '0':
            current += prev1
        
        two_digit = int(s[i-2:i])
        if 10 <= two_digit <= 26:
            current += prev2
        
        prev2 = prev1
        prev1 = current
    
    return prev1

# Example
print(num_decodings("12"))   # 2 ("AB" or "L")
print(num_decodings("226"))  # 3 ("BZ", "VF", "BBF")
```

---

## 11.6 DP Pattern 5: State Machine

### 11.6.1 Best Time to Buy and Sell Stock

```python
def max_profit(prices):
    """
    LeetCode 121: Best Time to Buy and Sell Stock
    
    One transaction (buy once, sell once).
    
    Track minimum price seen so far, maximize profit.
    """
    min_price = float('inf')
    max_profit = 0
    
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    
    return max_profit

# Time: O(n)
# Space: O(1)


def max_profit_k_transactions(k, prices):
    """
    LeetCode 188: Best Time to Buy and Sell Stock IV
    
    At most k transactions.
    
    State Machine:
    - buy[i][j] = max profit after j transactions, holding stock on day i
    - sell[i][j] = max profit after j transactions, not holding on day i
    
    Transitions:
    - buy[i][j] = max(buy[i-1][j], sell[i-1][j-1] - prices[i])
    - sell[i][j] = max(sell[i-1][j], buy[i-1][j] + prices[i])
    """
    if not prices or k == 0:
        return 0
    
    n = len(prices)
    
    # If k >= n/2, unlimited transactions
    if k >= n // 2:
        return sum(max(prices[i] - prices[i-1], 0) for i in range(1, n))
    
    # DP
    buy = [[-float('inf')] * (k + 1) for _ in range(n)]
    sell = [[0] * (k + 1) for _ in range(n)]
    
    for i in range(n):
        for j in range(1, k + 1):
            if i == 0:
                buy[i][j] = -prices[i]
                sell[i][j] = 0
            else:
                buy[i][j] = max(buy[i-1][j], sell[i-1][j-1] - prices[i])
                sell[i][j] = max(sell[i-1][j], buy[i-1][j] + prices[i])
    
    return sell[n-1][k]

# Time: O(n × k)
# Space: O(n × k)

# Example
print(max_profit_k_transactions(2, [3, 2, 6, 5, 0, 3]))  # 7
# Buy at 2, sell at 6, buy at 0, sell at 3
```

---

## 11.7 DP Pattern 6: Interval DP

### 11.7.1 Burst Balloons

```python
def max_coins(nums):
    """
    LeetCode 312: Burst Balloons
    
    Burst balloons to maximize coins.
    coins = nums[left] * nums[i] * nums[right]
    
    State: dp[left][right] = max coins bursting balloons in (left, right)
    Recurrence: Try each balloon i as the LAST one to burst
      dp[left][right] = max(
        dp[left][i] + dp[i][right] + 
        nums[left] * nums[i] * nums[right]
      ) for i in (left, right)
    
    Key insight: Think about which balloon to burst LAST,
    not first (avoids dependency issues)
    """
    # Add virtual balloons with value 1 at boundaries
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0] * n for _ in range(n)]
    
    # Iterate by interval length
    for length in range(2, n):
        for left in range(n - length):
            right = left + length
            
            # Try each balloon i as last to burst
            for i in range(left + 1, right):
                coins = nums[left] * nums[i] * nums[right]
                dp[left][right] = max(
                    dp[left][right],
                    dp[left][i] + dp[i][right] + coins
                )
    
    return dp[0][n-1]

# Time: O(n³)
# Space: O(n²)

# Example
print(max_coins([3, 1, 5, 8]))  # 167
```

---

## 11.8 Advanced DP Techniques

### 11.8.1 Bitmask DP

```python
def shortest_path_length(graph):
    """
    LeetCode 847: Shortest Path Visiting All Nodes
    
    Find shortest path that visits all nodes.
    Can revisit nodes and reuse edges.
    
    State: dp[node][mask] = shortest path ending at node with visited mask
    mask: bitmask representing visited nodes
    """
    from collections import deque
    
    n = len(graph)
    target = (1 << n) - 1  # All nodes visited
    
    # BFS with state (node, mask, distance)
    queue = deque([(i, 1 << i, 0) for i in range(n)])
    visited = {(i, 1 << i) for i in range(n)}
    
    while queue:
        node, mask, dist = queue.popleft()
        
        if mask == target:
            return dist
        
        for neighbor in graph[node]:
            new_mask = mask | (1 << neighbor)
            
            if (neighbor, new_mask) not in visited:
                visited.add((neighbor, new_mask))
                queue.append((neighbor, new_mask, dist + 1))
    
    return -1

# Time: O(n² × 2^n)
# Space: O(n × 2^n)
```

---

### 11.8.2 Digit DP

```python
def count_digit_one(n):
    """
    LeetCode 233: Number of Digit One
    
    Count occurrences of digit 1 in all numbers from 1 to n.
    
    Digit DP: Process digit by digit
    """
    def count_ones(position, count, tight):
        """
        position: current digit position
        count: number of 1s so far
        tight: whether we're bounded by n
        """
        if position == -1:
            return count
        
        limit = int(s[position]) if tight else 9
        result = 0
        
        for digit in range(limit + 1):
            new_count = count + (1 if digit == 1 else 0)
            new_tight = tight and (digit == limit)
            result += count_ones(position - 1, new_count, new_tight)
        
        return result
    
    s = str(n)
    return count_ones(len(s) - 1, 0, True)

# Time: O(log n × log n)
# Space: O(log n)
```

---

## Practice Questions

### Fill in the Gaps

1. Dynamic programming requires ________ substructure and ________ subproblems.
2. Memoization is a ________ approach while tabulation is ________.
3. The time complexity of 0/1 knapsack is ________.
4. LCS has time complexity ________ and space complexity ________.
5. To optimize space in DP, we only keep ________ states.

### True or False

1. DP always provides optimal solution. **[T/F]**
2. Memoization uses less space than tabulation. **[T/F]**
3. All DP problems can be solved with greedy algorithms. **[T/F]**
4. Bottom-up DP is always faster than top-down. **[T/F]**
5. Knapsack problem can be solved in polynomial time. **[T/F]**

### Multiple Choice

1. Which is NOT a characteristic of DP problems?
   - A) Optimal substructure
   - B) Overlapping subproblems
   - C) Greedy choice property
   - D) Memoization helps

2. Time complexity of Fibonacci with DP?
   - A) O(1)
   - B) O(n)
   - C) O(n log n)
   - D) O(2^n)

3. Space complexity of coin change problem?
   - A) O(1)
   - B) O(coins)
   - C) O(amount)
   - D) O(coins × amount)

### Code Challenge

```python
def max_product_subarray(nums):
    """
    LeetCode 152: Maximum Product Subarray
    
    Find contiguous subarray with largest product.
    
    Example: [2,3,-2,4] → 6 (subarray [2,3])
    Example: [-2,0,-1] → 0
    
    Hint: Track both max and min (negative × negative = positive)
    Use DP to solve.
    """
    # Your code here
    pass
```

---

## Answers

<details>
<summary><strong>View Answers</strong></summary>

### Fill in the Gaps

1. **optimal, overlapping**
2. **top-down, bottom-up**
3. **O(n × capacity)** or **O(nW)**
4. **O(m × n), O(m × n)**
5. **necessary** or **previous**

### True or False

1. **True** - DP finds optimal solution when problem has optimal substructure
2. **False** - Both use O(n) space; memoization also has recursion stack
3. **False** - DP handles overlapping subproblems; greedy makes local choices
4. **False** - Similar performance; bottom-up avoids recursion overhead
5. **True** - O(nW) is polynomial in n and W (pseudo-polynomial)

### Multiple Choice

1. **C** - Greedy choice is different from DP; DP considers all options
2. **B** - With memoization or tabulation
3. **C** - DP array of size amount

### Code Challenge Answer

```python
def max_product_subarray(nums):
    """
    Track max and min products ending at each position.
    Negative × negative can become maximum.
    """
    if not nums:
        return 0
    
    max_prod = min_prod = result = nums[0]
    
    for i in range(1, len(nums)):
        num = nums[i]
        
        # Current number could flip max/min if negative
        candidates = [num, max_prod * num, min_prod * num]
        max_prod = max(candidates)
        min_prod = min(candidates)
        
        result = max(result, max_prod)
    
    return result

# Time: O(n)
# Space: O(1)

# Example
print(max_product_subarray([2, 3, -2, 4]))  # 6
print(max_product_subarray([-2, 0, -1]))    # 0
print(max_product_subarray([-2, 3, -4]))    # 24 (entire array)
```

</details>

---

## LeetCode Problems (NeetCode.io)

### DP - Easy ✅
- 70. Climbing Stairs
- 746. Min Cost Climbing Stairs

### DP - Medium 🟨
- 5. Longest Palindromic Substring
- 22. Generate Parentheses
- 62. Unique Paths (IMPORTANT)
- 91. Decode Ways
- 139. Word Break (IMPORTANT)
- 152. Maximum Product Subarray
- 198. House Robber (IMPORTANT)
- 213. House Robber II
- 300. Longest Increasing Subsequence (IMPORTANT)
- 322. Coin Change (VERY IMPORTANT)
- 416. Partition Equal Subset Sum (IMPORTANT)
- 494. Target Sum
- 647. Palindromic Substrings
- 1143. Longest Common Subsequence (IMPORTANT)

### DP - Hard 🔴
- 10. Regular Expression Matching
- 72. Edit Distance (VERY IMPORTANT)
- 115. Distinct Subsequences
- 312. Burst Balloons
- 1000. Minimum Cost to Merge Stones

---

## Summary

### DP Pattern Recognition

**1D Array:**
- Fibonacci-like: dp[i] depends on previous few
- Examples: Climbing Stairs, House Robber, Decode Ways

**2D Grid:**
- Path counting: Unique Paths, Min Path Sum
- String comparison: LCS, Edit Distance

**Knapsack:**
- Subset selection: 0/1 Knapsack, Partition
- Each item: take or skip

**Subsequence:**
- Palindromes: LPS, Palindromic Substrings
- String matching: LCS

**State Machine:**
- Multiple states per step: Stock Trading
- Transitions between states

**Interval:**
- Subproblems on ranges: Burst Balloons
- Process intervals recursively

---

*Continue to: [12. Bit Manipulation →](12-bit-manipulation.md)*
