# 13. Advanced Topics

## 13.1 Greedy Algorithms

### 13.1.1 Greedy Algorithm Fundamentals

**Definition**: A greedy algorithm makes the locally optimal choice at each step with the hope of finding a global optimum. It never reconsiders its choices.

**Key Characteristics:**
1. **Greedy Choice Property**: Local optimum leads to global optimum
2. **Optimal Substructure**: Optimal solution contains optimal solutions to subproblems
3. **No Backtracking**: Once a choice is made, it's never reconsidered

**Greedy vs Dynamic Programming:**
```
Greedy:
- Makes one choice at each step
- Never looks back
- Faster (O(n) or O(n log n))
- Not always correct

DP:
- Considers all choices
- Builds solution from subproblems
- Slower (O(n²) or higher)
- Always correct (if applicable)
```

**When to Use Greedy:**
- Problem has greedy choice property
- Can prove correctness
- Optimization problems (max/min)
- Scheduling, intervals, selection

**Real-World Examples:**
- **Coin change**: Use largest coins first (only works for certain coin systems)
- **Huffman coding**: Build tree by combining smallest frequencies
- **Dijkstra's algorithm**: Always expand nearest unvisited node
- **Activity selection**: Choose earliest ending activity

---

### 13.1.2 Interval Problems

#### Jump Game

```python
def can_jump(nums):
    """
    LeetCode 55: Jump Game
    
    Can reach last index starting from first?
    nums[i] = maximum jump length from i.
    
    Greedy: Track maximum reachable position.
    """
    max_reach = 0
    
    for i in range(len(nums)):
        # Can't reach current position
        if i > max_reach:
            return False
        
        # Update max reachable
        max_reach = max(max_reach, i + nums[i])
        
        # Already can reach end
        if max_reach >= len(nums) - 1:
            return True
    
    return True

# Time: O(n)
# Space: O(1)

# Example
print(can_jump([2, 3, 1, 1, 4]))  # True
print(can_jump([3, 2, 1, 0, 4]))  # False


def jump(nums):
    """
    LeetCode 45: Jump Game II
    
    Minimum jumps to reach last index.
    
    Greedy: Use BFS-like approach, process level by level.
    """
    if len(nums) <= 1:
        return 0
    
    jumps = 0
    current_end = 0
    farthest = 0
    
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        
        # Reached end of current level
        if i == current_end:
            jumps += 1
            current_end = farthest
            
            # Can reach end
            if current_end >= len(nums) - 1:
                break
    
    return jumps

# Time: O(n)
# Space: O(1)

# Example
print(jump([2, 3, 1, 1, 4]))  # 2 (jump 1 step to index 1, then 3 steps to end)
```

---

#### Meeting Rooms

```python
def can_attend_meetings(intervals):
    """
    LeetCode 252: Meeting Rooms
    
    Check if person can attend all meetings (no overlaps).
    
    Greedy: Sort by start time, check consecutive overlaps.
    """
    if not intervals:
        return True
    
    intervals.sort(key=lambda x: x[0])
    
    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i-1][1]:
            return False
    
    return True

# Time: O(n log n)
# Space: O(1)


def min_meeting_rooms(intervals):
    """
    LeetCode 253: Meeting Rooms II
    
    Minimum number of meeting rooms needed.
    
    Greedy: Track all start and end times separately.
    """
    if not intervals:
        return 0
    
    starts = sorted([i[0] for i in intervals])
    ends = sorted([i[1] for i in intervals])
    
    rooms = 0
    max_rooms = 0
    s = e = 0
    
    while s < len(starts):
        if starts[s] < ends[e]:
            rooms += 1
            max_rooms = max(max_rooms, rooms)
            s += 1
        else:
            rooms -= 1
            e += 1
    
    return max_rooms

# Time: O(n log n)
# Space: O(n)

# Example
intervals = [[0,30], [5,10], [15,20]]
print(min_meeting_rooms(intervals))  # 2
```

---

#### Merge Intervals

```python
def merge_intervals(intervals):
    """
    LeetCode 56: Merge Intervals
    
    Merge overlapping intervals.
    
    Greedy: Sort, then merge consecutive overlapping intervals.
    """
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        
        if current[0] <= last[1]:
            # Overlapping, merge
            last[1] = max(last[1], current[1])
        else:
            # Non-overlapping, add new
            merged.append(current)
    
    return merged

# Time: O(n log n)
# Space: O(n)

# Example
intervals = [[1,3], [2,6], [8,10], [15,18]]
print(merge_intervals(intervals))  # [[1,6], [8,10], [15,18]]
```

---

### 13.1.3 Partitioning Problems

#### Partition Labels

```python
def partition_labels(s):
    """
    LeetCode 763: Partition Labels
    
    Partition string so each letter appears in at most one part.
    Maximize number of parts.
    
    Greedy: Track last occurrence of each character.
    """
    # Last occurrence of each character
    last = {c: i for i, c in enumerate(s)}
    
    result = []
    start = 0
    end = 0
    
    for i, char in enumerate(s):
        end = max(end, last[char])
        
        # Reached end of partition
        if i == end:
            result.append(end - start + 1)
            start = i + 1
    
    return result

# Time: O(n)
# Space: O(1) - at most 26 letters

# Example
print(partition_labels("ababcbacadefegdehijhklij"))
# [9, 7, 8] → "ababcbaca", "defegde", "hijhklij"
```

---

### 13.1.4 Huffman Coding

```python
import heapq
from collections import Counter

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

def huffman_coding(text):
    """
    Build Huffman tree for compression.
    
    Greedy: Always combine two nodes with lowest frequencies.
    """
    if not text:
        return {}
    
    # Count frequencies
    freq = Counter(text)
    
    # Build heap of nodes
    heap = [Node(char, f) for char, f in freq.items()]
    heapq.heapify(heap)
    
    # Build tree
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        
        heapq.heappush(heap, merged)
    
    # Generate codes
    root = heap[0]
    codes = {}
    
    def generate_codes(node, code):
        if node.char is not None:
            codes[node.char] = code
            return
        
        if node.left:
            generate_codes(node.left, code + '0')
        if node.right:
            generate_codes(node.right, code + '1')
    
    generate_codes(root, '')
    return codes

# Time: O(n log n)
# Space: O(n)

# Example
text = "huffman"
codes = huffman_coding(text)
print(codes)  # {'h': '00', 'u': '01', 'f': '10', 'm': '110', 'a': '111', 'n': '1111'}
```

---

## 13.2 Mathematical Algorithms

### 13.2.1 Number Theory

#### Greatest Common Divisor (GCD)

```python
def gcd(a, b):
    """
    Euclidean algorithm for GCD.
    
    gcd(a, b) = gcd(b, a % b)
    """
    while b:
        a, b = b, a % b
    return a

# Time: O(log min(a, b))
# Space: O(1)

# Recursive version:
def gcd_recursive(a, b):
    if b == 0:
        return a
    return gcd_recursive(b, a % b)

# Python built-in:
import math
print(math.gcd(48, 18))  # 6


def lcm(a, b):
    """Least Common Multiple"""
    return abs(a * b) // gcd(a, b)

# Example
print(gcd(48, 18))  # 6
print(lcm(12, 18))  # 36
```

---

#### Prime Numbers

```python
def is_prime(n):
    """
    Check if n is prime.
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check odd divisors up to √n
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    
    return True

# Time: O(√n)
# Space: O(1)


def sieve_of_eratosthenes(n):
    """
    Find all primes up to n.
    
    Sieve of Eratosthenes algorithm.
    """
    if n < 2:
        return []
    
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            # Mark multiples as not prime
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    return [i for i in range(n + 1) if is_prime[i]]

# Time: O(n log log n)
# Space: O(n)

# Example
print(is_prime(17))  # True
print(sieve_of_eratosthenes(30))  # [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def count_primes(n):
    """
    LeetCode 204: Count Primes
    
    Count primes less than n.
    """
    if n < 3:
        return 0
    
    is_prime = [True] * n
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n, i):
                is_prime[j] = False
    
    return sum(is_prime)

# Example
print(count_primes(10))  # 4 (2, 3, 5, 7)
```

---

#### Power and Modular Arithmetic

```python
def power(base, exp):
    """
    Calculate base^exp efficiently.
    
    Fast exponentiation using divide and conquer.
    """
    if exp == 0:
        return 1
    if exp == 1:
        return base
    
    half = power(base, exp // 2)
    
    if exp % 2 == 0:
        return half * half
    else:
        return base * half * half

# Time: O(log exp)
# Space: O(log exp) - recursion


def power_mod(base, exp, mod):
    """
    LeetCode 50: Pow(x, n) with modulo
    
    Calculate (base^exp) % mod efficiently.
    """
    result = 1
    base = base % mod
    
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        
        exp = exp >> 1
        base = (base * base) % mod
    
    return result

# Time: O(log exp)
# Space: O(1)

# Example
print(power(2, 10))  # 1024
print(power_mod(2, 10, 1000))  # 24 (1024 % 1000)
```

---

### 13.2.2 Combinatorics

#### Permutations and Combinations

```python
def factorial(n):
    """Calculate n!"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)


def permutations(n, r):
    """
    P(n, r) = n! / (n-r)!
    
    Number of ways to arrange r items from n items.
    """
    return factorial(n) // factorial(n - r)


def combinations(n, r):
    """
    C(n, r) = n! / (r! × (n-r)!)
    
    Number of ways to choose r items from n items.
    """
    return factorial(n) // (factorial(r) * factorial(n - r))


# Optimized combination (avoids large factorials)
def combinations_optimized(n, r):
    """Pascal's triangle approach"""
    if r > n - r:
        r = n - r
    
    result = 1
    for i in range(r):
        result = result * (n - i) // (i + 1)
    
    return result

# Example
print(permutations(5, 3))  # 60
print(combinations(5, 3))  # 10
```

---

#### Pascal's Triangle

```python
def generate_pascals_triangle(num_rows):
    """
    LeetCode 118: Pascal's Triangle
    
    Generate first num_rows of Pascal's triangle.
    
    Each number is sum of two numbers above it.
    """
    if num_rows == 0:
        return []
    
    triangle = [[1]]
    
    for i in range(1, num_rows):
        row = [1]
        
        for j in range(1, i):
            row.append(triangle[i-1][j-1] + triangle[i-1][j])
        
        row.append(1)
        triangle.append(row)
    
    return triangle

# Time: O(num_rows²)
# Space: O(num_rows²)

# Example
print(generate_pascals_triangle(5))
# [[1], [1,1], [1,2,1], [1,3,3,1], [1,4,6,4,1]]


def get_row(row_index):
    """
    LeetCode 119: Pascal's Triangle II
    
    Get specific row (0-indexed).
    
    Space optimized to O(k).
    """
    row = [1]
    
    for i in range(row_index):
        # Build next row from current
        row = [1] + [row[j] + row[j+1] for j in range(len(row)-1)] + [1]
    
    return row

# Example
print(get_row(3))  # [1, 3, 3, 1]
```

---

### 13.2.3 Geometry

#### Valid Square

```python
def valid_square(p1, p2, p3, p4):
    """
    LeetCode 593: Valid Square
    
    Check if 4 points form a square.
    
    Square has: 4 equal sides + 2 equal diagonals
    """
    def distance_squared(p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
    
    points = [p1, p2, p3, p4]
    distances = []
    
    # Calculate all pairwise distances
    for i in range(4):
        for j in range(i+1, 4):
            distances.append(distance_squared(points[i], points[j]))
    
    distances.sort()
    
    # Check: 4 equal sides and 2 equal diagonals
    return (distances[0] > 0 and
            distances[0] == distances[1] == distances[2] == distances[3] and
            distances[4] == distances[5])

# Example
print(valid_square([0,0], [1,1], [1,0], [0,1]))  # True
```

---

## 13.3 String Algorithms

### 13.3.1 String Matching

#### KMP Algorithm (Knuth-Morris-Pratt)

```python
def kmp_search(text, pattern):
    """
    KMP string matching algorithm.
    
    Find all occurrences of pattern in text.
    Preprocessing: Build LPS (Longest Proper Prefix which is also Suffix) array.
    """
    def build_lps(pattern):
        """Build longest proper prefix suffix array"""
        m = len(pattern)
        lps = [0] * m
        length = 0
        i = 1
        
        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        
        return lps
    
    n, m = len(text), len(pattern)
    lps = build_lps(pattern)
    
    result = []
    i = j = 0
    
    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
        
        if j == m:
            result.append(i - j)
            j = lps[j - 1]
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return result

# Time: O(n + m)
# Space: O(m)

# Example
text = "ababcababcabc"
pattern = "abc"
print(kmp_search(text, pattern))  # [2, 7, 10]
```

---

#### Rabin-Karp Algorithm

```python
def rabin_karp(text, pattern):
    """
    Rabin-Karp string matching using rolling hash.
    
    Use polynomial rolling hash with modulo.
    """
    n, m = len(text), len(pattern)
    if m > n:
        return []
    
    base = 256
    mod = 10**9 + 7
    
    # Hash pattern
    pattern_hash = 0
    for char in pattern:
        pattern_hash = (pattern_hash * base + ord(char)) % mod
    
    # Precompute base^(m-1) for rolling
    power = pow(base, m - 1, mod)
    
    # Rolling hash
    text_hash = 0
    result = []
    
    for i in range(n):
        # Add new character
        text_hash = (text_hash * base + ord(text[i])) % mod
        
        # Remove old character if window full
        if i >= m:
            text_hash = (text_hash - ord(text[i - m]) * power) % mod
        
        # Check if hashes match
        if i >= m - 1 and text_hash == pattern_hash:
            # Verify actual match (avoid hash collision)
            if text[i - m + 1:i + 1] == pattern:
                result.append(i - m + 1)
    
    return result

# Time: O(n + m) average, O(nm) worst
# Space: O(1)

# Example
print(rabin_karp("ababcababc", "abc"))  # [2, 7]
```

---

### 13.3.2 String Manipulation

#### Longest Common Prefix

```python
def longest_common_prefix(strs):
    """
    LeetCode 14: Longest Common Prefix
    
    Find longest common prefix string amongst array of strings.
    """
    if not strs:
        return ""
    
    # Start with first string
    prefix = strs[0]
    
    for s in strs[1:]:
        # Reduce prefix until it matches
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    
    return prefix

# Time: O(S) where S = sum of all characters
# Space: O(1)

# Alternative: Vertical scanning
def longest_common_prefix_v2(strs):
    """Compare character by character"""
    if not strs:
        return ""
    
    for i in range(len(strs[0])):
        char = strs[0][i]
        
        for s in strs[1:]:
            if i >= len(s) or s[i] != char:
                return strs[0][:i]
    
    return strs[0]

# Example
print(longest_common_prefix(["flower", "flow", "flight"]))  # "fl"
```

---

#### String Compression

```python
def compress(chars):
    """
    LeetCode 443: String Compression
    
    Compress in-place: ["a","a","b","b","c","c","c"]
    → ["a","2","b","2","c","3"]
    
    Return new length.
    """
    write = 0
    read = 0
    
    while read < len(chars):
        char = chars[read]
        count = 0
        
        # Count consecutive characters
        while read < len(chars) and chars[read] == char:
            read += 1
            count += 1
        
        # Write character
        chars[write] = char
        write += 1
        
        # Write count if > 1
        if count > 1:
            for digit in str(count):
                chars[write] = digit
                write += 1
    
    return write

# Time: O(n)
# Space: O(1)

# Example
chars = ["a","a","b","b","c","c","c"]
length = compress(chars)
print(chars[:length])  # ['a', '2', 'b', '2', 'c', '3']
```

---

#### Valid Parentheses Variations

```python
def is_valid(s):
    """
    LeetCode 20: Valid Parentheses
    
    Check if brackets are properly closed.
    """
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            top = stack.pop() if stack else '#'
            if mapping[char] != top:
                return False
        else:
            stack.append(char)
    
    return not stack

# Time: O(n)
# Space: O(n)


def min_add_to_make_valid(s):
    """
    LeetCode 921: Minimum Add to Make Parentheses Valid
    
    Return minimum insertions needed.
    """
    open_needed = 0
    close_needed = 0
    
    for char in s:
        if char == '(':
            close_needed += 1
        elif char == ')':
            if close_needed > 0:
                close_needed -= 1
            else:
                open_needed += 1
    
    return open_needed + close_needed

# Example
print(is_valid("()[]{}"))  # True
print(min_add_to_make_valid("())"))  # 1
```

---

### 13.3.3 Pattern Matching

#### Regular Expression Matching

```python
def is_match(s, p):
    """
    LeetCode 10: Regular Expression Matching
    
    Implement regex matching with '.' and '*'.
    '.' matches any single character
    '*' matches zero or more of preceding element
    
    DP approach.
    """
    m, n = len(s), len(p)
    
    # dp[i][j] = s[0:i] matches p[0:j]
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    # Handle patterns like a*, a*b*, etc. (match empty string)
    for j in range(2, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 2]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                # Don't use * (zero occurrences)
                dp[i][j] = dp[i][j - 2]
                
                # Use * (one or more occurrences)
                if p[j - 2] == '.' or p[j - 2] == s[i - 1]:
                    dp[i][j] = dp[i][j] or dp[i - 1][j]
            elif p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    
    return dp[m][n]

# Time: O(mn)
# Space: O(mn)

# Example
print(is_match("aa", "a"))     # False
print(is_match("aa", "a*"))    # True
print(is_match("ab", ".*"))    # True
```

---

#### Wildcard Matching

```python
def is_match_wildcard(s, p):
    """
    LeetCode 44: Wildcard Matching
    
    '?' matches any single character
    '*' matches any sequence (including empty)
    """
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    # Handle patterns like *, **, *a*, etc.
    for j in range(1, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 1]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                # Don't use * OR use *
                dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
            elif p[j - 1] == '?' or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    
    return dp[m][n]

# Time: O(mn)
# Space: O(mn)

# Example
print(is_match_wildcard("aa", "a"))   # False
print(is_match_wildcard("aa", "*"))   # True
print(is_match_wildcard("cb", "?a"))  # False
```

---

## Practice Questions

### Fill in the Gaps

1. Greedy algorithms make ________ optimal choices at each step.
2. The time complexity of the Euclidean algorithm for GCD is ________.
3. The Sieve of Eratosthenes has time complexity ________.
4. KMP algorithm preprocesses the ________ to build LPS array.
5. In Huffman coding, we combine nodes with ________ frequencies.

### True or False

1. Greedy algorithms always produce optimal solutions. **[T/F]**
2. Fast exponentiation has O(log n) time complexity. **[T/F]**
3. KMP is faster than naive string matching. **[T/F]**
4. GCD can be computed in polynomial time. **[T/F]**
5. Rabin-Karp has O(n) worst-case time. **[T/F]**

### Multiple Choice

1. Which is a greedy algorithm?
   - A) Dijkstra's
   - B) Bellman-Ford
   - C) Floyd-Warshall
   - D) All shortest path algorithms

2. Time to check if n is prime?
   - A) O(n)
   - B) O(√n)
   - C) O(log n)
   - D) O(1)

3. KMP preprocessing builds:
   - A) Suffix array
   - B) LPS array
   - C) Hash table
   - D) Trie

### Code Challenge

```python
def reorganize_string(s):
    """
    LeetCode 767: Reorganize String
    
    Rearrange string so no two adjacent characters are same.
    Return any valid rearrangement, or "" if impossible.
    
    Example: "aab" → "aba"
    Example: "aaab" → "" (impossible)
    
    Use greedy approach with heap.
    Hint: Always place most frequent character first (if valid).
    """
    # Your code here
    pass
```

---

## Answers

<details>
<summary><strong>View Answers</strong></summary>

### Fill in the Gaps

1. **locally** (or greedy)
2. **O(log min(a, b))**
3. **O(n log log n)**
4. **pattern**
5. **smallest** (or lowest)

### True or False

1. **False** - Only when greedy choice property holds
2. **True** - Divide exponent by 2 each time
3. **True** - KMP is O(n+m), naive is O(nm)
4. **True** - Euclidean algorithm is polynomial
5. **False** - O(nm) worst case due to hash collisions

### Multiple Choice

1. **A** - Dijkstra uses greedy approach
2. **B** - Check divisors up to √n
3. **B** - Longest Proper Prefix which is also Suffix

### Code Challenge Answer

```python
import heapq
from collections import Counter

def reorganize_string(s):
    """
    Use max heap to always pick most frequent character.
    Skip if same as previous.
    """
    freq = Counter(s)
    
    # Check if possible
    if max(freq.values()) > (len(s) + 1) // 2:
        return ""
    
    # Max heap (negate for max heap in Python)
    heap = [(-count, char) for char, count in freq.items()]
    heapq.heapify(heap)
    
    result = []
    prev_count, prev_char = 0, ''
    
    while heap:
        count, char = heapq.heappop(heap)
        result.append(char)
        
        # Add previous back if any left
        if prev_count < 0:
            heapq.heappush(heap, (prev_count, prev_char))
        
        # Update previous
        prev_count = count + 1  # Increment (was negative)
        prev_char = char
    
    return ''.join(result)

# Time: O(n log k) where k = unique characters
# Space: O(k)

# Example
print(reorganize_string("aab"))   # "aba" or "baa"
print(reorganize_string("aaab"))  # ""
```

</details>

---

## LeetCode Problems (NeetCode.io)

### Greedy - Easy ✅
- 121. Best Time to Buy and Sell Stock
- 455. Assign Cookies

### Greedy - Medium 🟨
- 45. Jump Game II
- 55. Jump Game (IMPORTANT)
- 56. Merge Intervals (IMPORTANT)
- 134. Gas Station
- 253. Meeting Rooms II
- 402. Remove K Digits
- 435. Non-overlapping Intervals
- 621. Task Scheduler
- 678. Valid Parenthesis String
- 763. Partition Labels (IMPORTANT)
- 846. Hand of Straights

### Greedy - Hard 🔴
- 135. Candy
- 358. Rearrange String k Distance Apart
- 767. Reorganize String

### Math - Easy ✅
- 66. Plus One
- 69. Sqrt(x)
- 202. Happy Number
- 204. Count Primes

### Math - Medium 🟨
- 2. Add Two Numbers
- 7. Reverse Integer
- 8. String to Integer (atoi)
- 29. Divide Two Integers
- 43. Multiply Strings
- 50. Pow(x, n)
- 166. Fraction to Recurring Decimal

### String - Medium 🟨
- 3. Longest Substring Without Repeating Characters
- 5. Longest Palindromic Substring
- 8. String to Integer (atoi)
- 14. Longest Common Prefix
- 49. Group Anagrams
- 151. Reverse Words in a String
- 271. Encode and Decode Strings

### String - Hard 🔴
- 10. Regular Expression Matching
- 44. Wildcard Matching
- 76. Minimum Window Substring

---

## Summary

### Greedy Patterns:
- **Intervals**: Sort + greedy selection
- **Scheduling**: Earliest deadline first
- **Partitioning**: Track boundaries

### Math Techniques:
- **GCD/LCM**: Euclidean algorithm
- **Primes**: Sieve of Eratosthenes
- **Power**: Fast exponentiation
- **Combinatorics**: Pascal's triangle

### String Algorithms:
- **Matching**: KMP O(n+m), Rabin-Karp
- **DP**: Edit distance, regex matching
- **Manipulation**: Two pointers, sliding window

---

*Continue to: [14. Interview Strategies →](14-interview-strategies.md)*