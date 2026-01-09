# 9. Hashing

## 9.1 Hash Table Fundamentals

### 9.1.1 Hash Table Definition

**Definition**: A hash table (hash map) is a data structure that maps keys to values using a **hash function**. It provides average O(1) time for insert, delete, and search operations by computing an index (hash) from the key to determine where to store/retrieve the value.

**Core Concepts:**

1. **Hash Function**: Converts key → integer index
   - `hash(key) → index in array`
   - Should distribute keys uniformly
   - Deterministic (same key → same hash)

2. **Hash Table**: Array + hash function
   - Array of buckets/slots
   - Each bucket stores key-value pairs
   - Index = hash(key) % array_size

3. **Collision**: Different keys hash to same index
   - Inevitable by pigeonhole principle
   - Need collision resolution strategy

**Visual Representation**:
```
Hash Table (size 7):

key: "apple" → hash("apple") = 345 → 345 % 7 = 2
key: "banana" → hash("banana") = 982 → 982 % 7 = 2  (collision!)

Array:
Index:  0    1    2              3    4    5    6
       [ ] [ ] ["apple"       ] [ ] [ ] [ ] [ ]
                ["banana"]
                    ↑
              Collision handled by chaining
```

**Why Hash Tables:**
- Insert: O(1) average
- Delete: O(1) average  
- Search: O(1) average
- Fast lookups by key
- Flexible key types

**Real-World Analogies**:
- **Phone book**: Name (key) → phone number (value)
- **Library catalog**: ISBN (key) → book location (value)
- **Dictionary**: Word (key) → definition (value)
- **Cache**: URL (key) → cached page (value)

**Real-World Uses**:
- Database indexing
- Caching (Redis, Memcached)
- Symbol tables in compilers
- Routers (IP → MAC address)
- Password storage (hashed)

---

### 9.1.2 Hash Functions

**Definition**: A hash function converts a key into an array index.

**Properties of Good Hash Function:**
1. **Deterministic**: Same input → same output
2. **Uniform distribution**: Minimizes collisions
3. **Fast to compute**: O(1) time
4. **Avalanche effect**: Small input change → big hash change

**Common Hash Functions:**

```python
# 1. Division Method (simple but not best)
def hash_division(key, table_size):
    """
    hash(key) = key % table_size
    
    Works for integers. Table size should be prime.
    """
    return key % table_size

# Example
print(hash_division(123, 10))  # 3
print(hash_division(456, 10))  # 6


# 2. Multiplication Method
def hash_multiplication(key, table_size):
    """
    hash(key) = floor(table_size * (key * A % 1))
    
    A is constant (golden ratio ≈ 0.618 works well)
    """
    A = 0.6180339887  # (√5 - 1) / 2
    return int(table_size * ((key * A) % 1))


# 3. String Hashing (Polynomial rolling hash)
def hash_string(s, table_size):
    """
    hash(s) = (s[0]*p^0 + s[1]*p^1 + ... + s[n-1]*p^(n-1)) % table_size
    
    p is prime (usually 31 or 37 for strings)
    """
    p = 31
    hash_value = 0
    p_pow = 1
    
    for char in s:
        hash_value = (hash_value + ord(char) * p_pow) % table_size
        p_pow = (p_pow * p) % table_size
    
    return hash_value

print(hash_string("hello", 100))  # Some value < 100


# 4. Python's built-in hash
def hash_builtin(key, table_size):
    """Python's hash() function"""
    return hash(key) % table_size

print(hash_builtin("apple", 10))
print(hash_builtin(123, 10))
```

---

### 9.1.3 Collision Resolution

**Problem**: Different keys hash to same index.

**Solution 1: Chaining (Separate Chaining)**

**Definition**: Each bucket contains a linked list of all key-value pairs that hash to that index.

```python
class HashTableChaining:
    """
    Hash table using chaining for collision resolution.
    
    Each bucket is a list of (key, value) pairs.
    """
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(size)]
    
    def _hash(self, key):
        """Hash function"""
        return hash(key) % self.size
    
    def insert(self, key, value):
        """
        Insert key-value pair.
        If key exists, update value.
        """
        index = self._hash(key)
        bucket = self.table[index]
        
        # Check if key exists, update if so
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        
        # Add new pair
        bucket.append((key, value))
    
    def get(self, key):
        """
        Get value for key.
        Returns None if not found.
        """
        index = self._hash(key)
        bucket = self.table[index]
        
        for k, v in bucket:
            if k == key:
                return v
        
        return None
    
    def delete(self, key):
        """Delete key-value pair"""
        index = self._hash(key)
        bucket = self.table[index]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                return True
        
        return False

# Time Complexity:
# - Average: O(1) for insert/get/delete
# - Worst: O(n) if all keys hash to same bucket
# 
# Space: O(n + m) where n = elements, m = table size

# Example usage
ht = HashTableChaining()
ht.insert("apple", 5)
ht.insert("banana", 7)
ht.insert("cherry", 3)
print(ht.get("apple"))   # 5
print(ht.get("grape"))   # None
ht.delete("banana")
print(ht.get("banana"))  # None
```

---

**Solution 2: Open Addressing**

**Definition**: Store all elements in the array itself. On collision, probe for next available slot.

**Linear Probing**:
```python
class HashTableLinearProbing:
    """
    Hash table using linear probing.
    
    On collision, check next slot: (hash + 1) % size
    Continue until empty slot found.
    """
    def __init__(self, size=10):
        self.size = size
        self.keys = [None] * size
        self.values = [None] * size
    
    def _hash(self, key):
        """Hash function"""
        return hash(key) % self.size
    
    def insert(self, key, value):
        """Insert with linear probing"""
        index = self._hash(key)
        
        # Linear probe until empty or matching key
        while self.keys[index] is not None:
            if self.keys[index] == key:
                # Update existing
                self.values[index] = value
                return
            
            # Move to next slot
            index = (index + 1) % self.size
        
        # Insert at empty slot
        self.keys[index] = key
        self.values[index] = value
    
    def get(self, key):
        """Get value with linear probing"""
        index = self._hash(key)
        
        # Probe until found or empty slot
        while self.keys[index] is not None:
            if self.keys[index] == key:
                return self.values[index]
            
            index = (index + 1) % self.size
        
        return None
    
    def delete(self, key):
        """
        Delete with linear probing.
        Mark slot as deleted (not None) to maintain probe chains.
        """
        index = self._hash(key)
        
        while self.keys[index] is not None:
            if self.keys[index] == key:
                self.keys[index] = "DELETED"  # Tombstone
                self.values[index] = None
                return True
            
            index = (index + 1) % self.size
        
        return False

# Time: O(1) average, O(n) worst if table nearly full
# Space: O(m) where m = table size

# Problems with linear probing:
# - Primary clustering: consecutive filled slots
# - Performance degrades as load factor increases
```

**Quadratic Probing**: Probe at quadratic intervals
```python
# Instead of (hash + 1), (hash + 2), ...
# Use (hash + 1²), (hash + 2²), (hash + 3²), ...

def quadratic_probe(hash_val, i, size):
    """Quadratic probing: (hash + i²) % size"""
    return (hash_val + i * i) % size

# Reduces primary clustering but has secondary clustering
```

**Double Hashing**: Use second hash function for probing
```python
def double_hash_probe(key, i, size):
    """
    Use two hash functions.
    probe(i) = (hash1(key) + i * hash2(key)) % size
    """
    hash1 = hash(key) % size
    hash2 = 1 + (hash(key) % (size - 1))  # Must be ≠ 0
    return (hash1 + i * hash2) % size

# Best open addressing method - no clustering
```

---

### 9.1.4 Load Factor and Rehashing

**Load Factor**: α = n / m (elements / table size)

**Guidelines:**
- Chaining: α can be > 1
- Open addressing: Keep α < 0.7 for good performance
- When α exceeds threshold → rehash

**Rehashing**:
```python
class HashTableWithRehash:
    """Hash table that automatically rehashes when load factor high"""
    
    def __init__(self, initial_size=10):
        self.size = initial_size
        self.count = 0
        self.table = [[] for _ in range(self.size)]
        self.load_factor_threshold = 0.75
    
    def _hash(self, key):
        return hash(key) % self.size
    
    def _rehash(self):
        """
        Rehash when load factor exceeds threshold.
        Typically double the size.
        """
        old_table = self.table
        self.size *= 2
        self.table = [[] for _ in range(self.size)]
        self.count = 0
        
        # Reinsert all elements
        for bucket in old_table:
            for key, value in bucket:
                self.insert(key, value)
    
    def insert(self, key, value):
        # Check load factor
        if self.count / self.size >= self.load_factor_threshold:
            self._rehash()
        
        index = self._hash(key)
        bucket = self.table[index]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        
        bucket.append((key, value))
        self.count += 1

# Rehashing time: O(n) but amortized O(1) per operation
```

---

## 9.2 Python Hash Tables

### 9.2.1 Python Dictionary

**Definition**: Python's built-in `dict` is an optimized hash table.

```python
# Creating dictionaries
d1 = {}  # Empty
d2 = {"apple": 5, "banana": 7}  # Literals
d3 = dict(apple=5, banana=7)  # Constructor
d4 = dict([("apple", 5), ("banana", 7)])  # From list

# Basic operations - all O(1) average
d = {}

# Insert/Update
d["apple"] = 5
d["banana"] = 7
d.update({"cherry": 3, "date": 9})

# Access
value = d["apple"]  # 5
value = d.get("grape")  # None (doesn't raise error)
value = d.get("grape", 0)  # 0 (default value)

# Delete
del d["banana"]
value = d.pop("cherry")  # Remove and return value
d.clear()  # Remove all

# Check existence
if "apple" in d:  # O(1)
    print("Found")

# Size
len(d)  # O(1)

# Iteration
for key in d:  # Iterate keys
    print(key, d[key])

for key, value in d.items():  # Iterate pairs
    print(key, value)

for value in d.values():  # Iterate values
    print(value)
```

---

### 9.2.2 Python Set

**Definition**: A set is an unordered collection of unique elements, implemented as a hash table.

```python
# Creating sets
s1 = set()  # Empty
s2 = {1, 2, 3}  # Literals
s3 = set([1, 2, 3, 2])  # From list (duplicates removed)

# Basic operations - all O(1) average
s = set()

# Add
s.add(1)
s.add(2)
s.add(1)  # No effect (already exists)

# Remove
s.remove(2)  # Raises KeyError if not found
s.discard(3)  # No error if not found
s.pop()  # Remove arbitrary element

# Check membership
if 1 in s:  # O(1)
    print("Found")

# Set operations
a = {1, 2, 3}
b = {2, 3, 4}

# Union: a ∪ b
union = a | b  # {1, 2, 3, 4}
union = a.union(b)

# Intersection: a ∩ b
intersection = a & b  # {2, 3}
intersection = a.intersection(b)

# Difference: a - b
diff = a - b  # {1}
diff = a.difference(b)

# Symmetric difference: a Δ b
sym_diff = a ^ b  # {1, 4}
sym_diff = a.symmetric_difference(b)

# Subset/Superset
is_subset = a <= b  # False
is_superset = a >= b  # False
```

---

### 9.2.3 Counter and defaultdict

**Counter**: Hash table for counting elements

```python
from collections import Counter

# Create counter
text = "mississippi"
counter = Counter(text)
# Counter({'i': 4, 's': 4, 'p': 2, 'm': 1})

# Most common
counter.most_common(2)  # [('i', 4), ('s', 4)]

# Operations
c1 = Counter(['a', 'b', 'c', 'a', 'b', 'b'])
c2 = Counter(['a', 'b', 'd'])

c1 + c2  # Add counts
c1 - c2  # Subtract (keep positive)
c1 & c2  # Intersection (min)
c1 | c2  # Union (max)
```

**defaultdict**: Dictionary with default values

```python
from collections import defaultdict

# Default value is int() = 0
dd = defaultdict(int)
dd['apple'] += 1  # No KeyError, starts at 0
dd['apple'] += 1
# dd = {'apple': 2}

# Default value is list
dd = defaultdict(list)
dd['fruits'].append('apple')
dd['fruits'].append('banana')
# dd = {'fruits': ['apple', 'banana']}

# Default value is set
dd = defaultdict(set)
dd['numbers'].add(1)
dd['numbers'].add(2)
# dd = {'numbers': {1, 2}}

# Custom default
dd = defaultdict(lambda: "Not found")
print(dd['missing'])  # "Not found"
```

---

## 9.3 Hash Table Patterns

### 9.3.1 Two Sum Pattern

**Pattern**: Find pairs/complements using hash table.

```python
def two_sum(nums, target):
    """
    LeetCode 1: Two Sum
    
    Find indices of two numbers that add to target.
    
    Approach: Store seen numbers in hash table.
    For each number, check if complement exists.
    """
    seen = {}  # value → index
    
    for i, num in enumerate(nums):
        complement = target - num
        
        if complement in seen:
            return [seen[complement], i]
        
        seen[num] = i
    
    return []

# Time: O(n)
# Space: O(n)

# Example
nums = [2, 7, 11, 15]
target = 9
print(two_sum(nums, target))  # [0, 1]


def two_sum_all_pairs(nums, target):
    """
    Find all unique pairs that sum to target.
    """
    seen = set()
    result = set()
    
    for num in nums:
        complement = target - num
        
        if complement in seen:
            # Add as sorted tuple to avoid duplicates
            pair = tuple(sorted([num, complement]))
            result.add(pair)
        
        seen.add(num)
    
    return list(result)

# Example
nums = [1, 5, 3, 7, 2, 8]
target = 10
print(two_sum_all_pairs(nums, target))  # [(2, 8), (3, 7)]
```

---

### 9.3.2 Frequency Counting Pattern

**Pattern**: Count occurrences using hash table.

```python
def group_anagrams(strs):
    """
    LeetCode 49: Group Anagrams
    
    Group strings that are anagrams of each other.
    
    Approach: Use sorted string as key.
    """
    from collections import defaultdict
    
    groups = defaultdict(list)
    
    for s in strs:
        # Sort string to get key
        key = ''.join(sorted(s))
        groups[key].append(s)
    
    return list(groups.values())

# Time: O(n * k log k) where k = max string length
# Space: O(n * k)

# Example
strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
print(group_anagrams(strs))
# [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]


def top_k_frequent(nums, k):
    """
    LeetCode 347: Top K Frequent Elements
    
    Return k most frequent elements.
    """
    from collections import Counter
    
    # Count frequencies
    count = Counter(nums)
    
    # Get k most common
    return [num for num, freq in count.most_common(k)]

# Time: O(n log k)
# Space: O(n)

# Example
nums = [1, 1, 1, 2, 2, 3]
k = 2
print(top_k_frequent(nums, k))  # [1, 2]
```

---

### 9.3.3 Sliding Window with Hash Pattern

```python
def longest_substring_without_repeating(s):
    """
    LeetCode 3: Longest Substring Without Repeating Characters
    
    Find length of longest substring with unique characters.
    
    Approach: Sliding window with hash table tracking last position.
    """
    char_index = {}  # char → last index
    max_length = 0
    start = 0
    
    for end, char in enumerate(s):
        # If char seen in current window, move start
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        
        # Update last position
        char_index[char] = end
        
        # Update max length
        max_length = max(max_length, end - start + 1)
    
    return max_length

# Time: O(n)
# Space: O(min(n, m)) where m = charset size

# Example
s = "abcabcbb"
print(longest_substring_without_repeating(s))  # 3 ("abc")


def min_window_substring(s, t):
    """
    LeetCode 76: Minimum Window Substring
    
    Find smallest substring of s containing all characters of t.
    
    Approach: Sliding window with two hash tables.
    """
    from collections import Counter
    
    if not t or not s:
        return ""
    
    # Count characters needed
    need = Counter(t)
    required = len(need)
    
    # Sliding window
    left = 0
    formed = 0  # Unique chars satisfied
    window_counts = {}
    
    # Result: (length, left, right)
    result = float('inf'), None, None
    
    for right, char in enumerate(s):
        # Add character to window
        window_counts[char] = window_counts.get(char, 0) + 1
        
        # Check if frequency matches
        if char in need and window_counts[char] == need[char]:
            formed += 1
        
        # Try to contract window
        while left <= right and formed == required:
            char = s[left]
            
            # Update result if smaller
            if right - left + 1 < result[0]:
                result = (right - left + 1, left, right)
            
            # Remove from window
            window_counts[char] -= 1
            if char in need and window_counts[char] < need[char]:
                formed -= 1
            
            left += 1
    
    return "" if result[0] == float('inf') else s[result[1]:result[2] + 1]

# Time: O(|s| + |t|)
# Space: O(|s| + |t|)

# Example
s = "ADOBECODEBANC"
t = "ABC"
print(min_window_substring(s, t))  # "BANC"
```

---

### 9.3.4 Hash Set for Deduplication

```python
def contains_duplicate(nums):
    """
    LeetCode 217: Contains Duplicate
    
    Check if array has duplicates.
    """
    return len(nums) != len(set(nums))

# Time: O(n)
# Space: O(n)

# Example
print(contains_duplicate([1, 2, 3, 1]))  # True
print(contains_duplicate([1, 2, 3, 4]))  # False


def longest_consecutive(nums):
    """
    LeetCode 128: Longest Consecutive Sequence
    
    Find length of longest consecutive sequence.
    
    Approach: Use set for O(1) lookup.
    Only start counting from sequence starts (no num-1 in set).
    """
    num_set = set(nums)
    max_length = 0
    
    for num in num_set:
        # Only start from sequence beginning
        if num - 1 not in num_set:
            current = num
            length = 1
            
            # Count consecutive numbers
            while current + 1 in num_set:
                current += 1
                length += 1
            
            max_length = max(max_length, length)
    
    return max_length

# Time: O(n) - each number visited at most twice
# Space: O(n)

# Example
nums = [100, 4, 200, 1, 3, 2]
print(longest_consecutive(nums))  # 4 ([1, 2, 3, 4])
```

---

### 9.3.5 Hash Map for Caching/Memoization

```python
def subarray_sum_equals_k(nums, k):
    """
    LeetCode 560: Subarray Sum Equals K
    
    Count subarrays with sum equal to k.
    
    Approach: Prefix sum with hash table.
    If prefix_sum[j] - prefix_sum[i] = k, then sum(i+1...j) = k.
    """
    from collections import defaultdict
    
    prefix_sum = 0
    count = 0
    sum_count = defaultdict(int)
    sum_count[0] = 1  # Empty prefix
    
    for num in nums:
        prefix_sum += num
        
        # Check if (prefix_sum - k) exists
        if prefix_sum - k in sum_count:
            count += sum_count[prefix_sum - k]
        
        # Add current prefix sum
        sum_count[prefix_sum] += 1
    
    return count

# Time: O(n)
# Space: O(n)

# Example
nums = [1, 1, 1]
k = 2
print(subarray_sum_equals_k(nums, k))  # 2 ([1,1], [1,1])


def clone_graph(node):
    """
    LeetCode 133: Clone Graph
    
    Deep copy of graph using hash map to track clones.
    """
    if not node:
        return None
    
    # Map: original → clone
    clones = {}
    
    def dfs(original):
        if original in clones:
            return clones[original]
        
        # Create clone
        clone = Node(original.val)
        clones[original] = clone
        
        # Clone neighbors
        for neighbor in original.neighbors:
            clone.neighbors.append(dfs(neighbor))
        
        return clone
    
    return dfs(node)

# Time: O(n + e) where e = edges
# Space: O(n)
```

---

## 9.4 Advanced Hashing Concepts

### 9.4.1 Rolling Hash

**Definition**: Hash that can be updated in O(1) when sliding a window.

```python
def rabin_karp(text, pattern):
    """
    Rabin-Karp string matching using rolling hash.
    
    Find all occurrences of pattern in text.
    """
    if not text or not pattern:
        return []
    
    n, m = len(text), len(pattern)
    if m > n:
        return []
    
    # Hash parameters
    base = 256  # Number of characters
    mod = 10**9 + 7
    
    # Compute hash of pattern
    pattern_hash = 0
    for char in pattern:
        pattern_hash = (pattern_hash * base + ord(char)) % mod
    
    # Compute base^(m-1) for removal
    power = pow(base, m - 1, mod)
    
    # Rolling hash for text
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

# Time: O(n + m) average, O(nm) worst case
# Space: O(1)

# Example
text = "ababcababc"
pattern = "abc"
print(rabin_karp(text, pattern))  # [2, 7]
```

---

### 9.4.2 Consistent Hashing

**Definition**: Hashing scheme that minimizes remapping when hash table resizes.

**Use Case**: Distributed systems, load balancing, caching

```python
import bisect
import hashlib

class ConsistentHash:
    """
    Consistent hashing for distributed systems.
    
    Maps keys to nodes such that adding/removing nodes
    only affects 1/n fraction of keys.
    """
    def __init__(self, nodes=None, virtual_nodes=100):
        self.virtual_nodes = virtual_nodes
        self.ring = {}  # hash → node
        self.sorted_keys = []
        
        if nodes:
            for node in nodes:
                self.add_node(node)
    
    def _hash(self, key):
        """Hash function"""
        return int(hashlib.md5(str(key).encode()).hexdigest(), 16)
    
    def add_node(self, node):
        """Add node to ring"""
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:{i}"
            hash_val = self._hash(virtual_key)
            self.ring[hash_val] = node
            bisect.insort(self.sorted_keys, hash_val)
    
    def remove_node(self, node):
        """Remove node from ring"""
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:{i}"
            hash_val = self._hash(virtual_key)
            del self.ring[hash_val]
            self.sorted_keys.remove(hash_val)
    
    def get_node(self, key):
        """Get node responsible for key"""
        if not self.ring:
            return None
        
        hash_val = self._hash(key)
        
        # Find first node clockwise
        idx = bisect.bisect(self.sorted_keys, hash_val)
        if idx == len(self.sorted_keys):
            idx = 0
        
        return self.ring[self.sorted_keys[idx]]

# Example usage
ch = ConsistentHash(['server1', 'server2', 'server3'])
print(ch.get_node('user123'))  # Maps to some server
print(ch.get_node('user456'))  # Maps to some server

# Add server - only ~1/4 of keys remapped
ch.add_node('server4')
```

---

## 9.5 Common Pitfalls

### 9.5.1 Mutable Keys

**Problem**: Using mutable objects as dictionary keys

```python
# BAD: List is mutable
# d = {[1, 2]: "value"}  # TypeError

# GOOD: Use tuple instead
d = {(1, 2): "value"}  # OK

# BAD: Set is mutable
# d = {{1, 2}: "value"}  # TypeError

# GOOD: Use frozenset
d = {frozenset([1, 2]): "value"}  # OK
```

---

### 9.5.2 Hash Collision Attacks

**Problem**: Malicious input causing many collisions

```python
# If attacker knows hash function, can craft keys
# that all hash to same bucket → O(n) operations

# Solution: Use randomized hashing (Python 3.3+)
# Hash function includes random seed per interpreter run
```

---

### 9.5.3 Iteration During Modification

**Problem**: Modifying dict/set while iterating

```python
# BAD: RuntimeError
d = {'a': 1, 'b': 2, 'c': 3}
# for key in d:
#     if key == 'b':
#         del d[key]  # Error!

# GOOD: Iterate over copy
for key in list(d.keys()):
    if key == 'b':
        del d[key]  # OK

# GOOD: Build new dict
d = {k: v for k, v in d.items() if k != 'b'}
```

---

## Practice Questions

### Fill in the Gaps

1. The average time complexity for hash table operations is ________.
2. A good hash function should distribute keys ________.
3. The load factor is defined as ________.
4. In chaining, each bucket contains a ________ of key-value pairs.
5. Python's built-in ________ function computes hash values.

### True or False

1. Hash tables guarantee O(1) worst-case lookup time. **[T/F]**
2. Sets in Python are implemented using hash tables. **[T/F]**
3. Dictionary keys must be immutable in Python. **[T/F]**
4. Open addressing uses linked lists for collisions. **[T/F]**
5. Rehashing is required when load factor gets too high. **[T/F]**

### Multiple Choice

1. Best collision resolution for high load factors?
   - A) Linear probing
   - B) Chaining
   - C) No collision handling needed
   - D) Quadratic probing

2. Time to build Counter from array of n elements?
   - A) O(1)
   - B) O(log n)
   - C) O(n)
   - D) O(n²)

3. Which is NOT hashable in Python?
   - A) Tuple
   - B) String
   - C) List
   - D) Integer

### Code Challenge

```python
def isomorphic_strings(s, t):
    """
    LeetCode 205: Isomorphic Strings
    
    Check if strings are isomorphic (characters can be mapped 1-to-1).
    
    Example: "egg", "add" → True (e→a, g→d)
    Example: "foo", "bar" → False (o maps to both a and r)
    
    Use hash tables to track mappings.
    """
    # Your code here
    pass
```

---

## Answers

<details>
<summary><strong>View Answers</strong></summary>

### Fill in the Gaps

1. **O(1)**
2. **uniformly** (or evenly)
3. **n / m** or **elements / table size**
4. **list** (or linked list)
5. **hash()**

### True or False

1. **False** - Average O(1), worst O(n) with collisions
2. **True** - Sets use hash tables for O(1) membership
3. **True** - Keys must be hashable (immutable)
4. **False** - Open addressing stores in array, chaining uses lists
5. **True** - Rehashing maintains good performance

### Multiple Choice

1. **B** - Chaining handles high loads better than probing
2. **C** - Must process each element once
3. **C** - Lists are mutable, not hashable

### Code Challenge Answer

```python
def isomorphic_strings(s, t):
    if len(s) != len(t):
        return False
    
    s_to_t = {}
    t_to_s = {}
    
    for c1, c2 in zip(s, t):
        # Check s → t mapping
        if c1 in s_to_t:
            if s_to_t[c1] != c2:
                return False
        else:
            s_to_t[c1] = c2
        
        # Check t → s mapping (must be bijective)
        if c2 in t_to_s:
            if t_to_s[c2] != c1:
                return False
        else:
            t_to_s[c2] = c1
    
    return True

# Time: O(n)
# Space: O(1) - at most 26 letters

# Examples
print(isomorphic_strings("egg", "add"))  # True
print(isomorphic_strings("foo", "bar"))  # False
print(isomorphic_strings("paper", "title"))  # True
```

</details>

---

## LeetCode Problems (NeetCode.io)

### Hash Tables - Easy ✅
- 1. Two Sum (VERY IMPORTANT)
- 217. Contains Duplicate
- 242. Valid Anagram
- 383. Ransom Note
- 387. First Unique Character in a String
- 389. Find the Difference

### Hash Tables - Medium 🟨
- 3. Longest Substring Without Repeating Characters (IMPORTANT)
- 49. Group Anagrams (IMPORTANT)
- 128. Longest Consecutive Sequence (IMPORTANT)
- 149. Max Points on a Line
- 205. Isomorphic Strings
- 347. Top K Frequent Elements (IMPORTANT)
- 380. Insert Delete GetRandom O(1)
- 454. 4Sum II
- 560. Subarray Sum Equals K (IMPORTANT)
- 692. Top K Frequent Words

### Hash Tables - Hard 🔴
- 76. Minimum Window Substring (VERY IMPORTANT)
- 149. Max Points on a Line

---

## Summary

### Hash Table Quick Reference

**When to use hash tables:**
- Need O(1) average lookup/insert/delete
- Counting frequencies
- Finding duplicates
- Two sum / complement problems
- Grouping/categorizing data
- Caching/memoization

**Key patterns:**
- **Two Sum**: Store complements
- **Frequency**: Counter for counting
- **Anagrams**: Sorted string as key
- **Sliding Window**: Track characters in window
- **Prefix Sum**: Map sum → index

**Python collections:**
- `dict`: General hash table
- `set`: Unique elements
- `Counter`: Frequency counting
- `defaultdict`: Default values

---

*Continue to: [10. Graphs →](10-graphs.md)*