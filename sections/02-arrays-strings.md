# 2. Arrays & Strings

## 2.1 Arrays

### 2.1.1 Array Fundamentals

**Definition**: An array is a contiguous block of memory that stores elements of the same data type. Each element can be accessed directly using its index (position number), which starts at 0 in most programming languages.

**Key Characteristics:**
- **Contiguous memory**: All elements stored next to each other in memory
- **Fixed size**: Size determined at creation (in most languages)
- **Direct access**: Can access any element in O(1) time using index
- **Same type**: All elements must be the same data type

**Real-World Analogy**: Arrays are like reserved parking spots in a lot - each spot has a number (index), and you can instantly drive to spot #5 without checking spots 1-4.

**Key Properties:**
- **Fixed size** (in most languages): Like a parking lot with 100 spots
- **Contiguous memory**: All spots are in a row
- **O(1) access**: Jump directly to any index
- **O(n) search**: Must check each element (unless sorted)

```python
# E-commerce: Product inventory array
class ProductInventory:
    """
    Manage product inventory using array.
    Like warehouse with numbered shelves.
    """
    def __init__(self, capacity):
        self.products = [None] * capacity  # Fixed-size array
        self.size = 0
    
    def add_product(self, product):
        """Add product - O(1) if space available"""
        if self.size < len(self.products):
            self.products[self.size] = product
            self.size += 1
            return True
        return False  # Array full
    
    def get_product(self, index):
        """Get product by index - O(1)"""
        if 0 <= index < self.size:
            return self.products[index]
        return None
    
    def find_product(self, product_id):
        """Find product by ID - O(n)"""
        for i in range(self.size):
            if self.products[i] and self.products[i]['id'] == product_id:
                return i
        return -1
    
    def remove_product(self, index):
        """Remove product - O(n) due to shifting"""
        if 0 <= index < self.size:
            # Shift all elements left
            for i in range(index, self.size - 1):
                self.products[i] = self.products[i + 1]
            self.products[self.size - 1] = None
            self.size -= 1
            return True
        return False

# Example usage
inventory = ProductInventory(100)
inventory.add_product({'id': 'LAPTOP-001', 'name': 'MacBook Pro', 'price': 1999})
inventory.add_product({'id': 'PHONE-001', 'name': 'iPhone 15', 'price': 999})

# O(1) access
product = inventory.get_product(0)  # MacBook Pro

# O(n) search
index = inventory.find_product('PHONE-001')  # Returns 1

# O(n) removal (requires shifting)
inventory.remove_product(0)  # Remove MacBook, iPhone shifts to index 0
```

### Array Operations Complexity

| Operation | Unsorted Array | Sorted Array | Notes |
|-----------|---------------|--------------|-------|
| Access by index | O(1) | O(1) | Direct memory access |
| Search | O(n) | O(log n) | Binary search if sorted |
| Insert at end | O(1) | O(n) | May need resorting |
| Insert at position | O(n) | O(n) | Requires shifting |
| Delete | O(n) | O(n) | Requires shifting |
| Find min/max | O(n) | O(1) | Ends of sorted array |

### 2.1.2 Common Array Problems

#### Problem 1: Two Sum (Hash Map Approach)

```python
def two_sum_products(prices, target_budget):
    """
    Find two products with prices summing to target budget.
    
    Real-world: Customer has $500, find 2 items totaling exactly $500.
    
    Approach: Use hash map to store seen prices.
    """
    seen = {}  # price -> index
    
    for i, product in enumerate(prices):
        price = product['price']
        needed = target_budget - price
        
        if needed in seen:
            return (seen[needed], i)  # Indices of the two products
        
        seen[price] = i
    
    return None

# Time: O(n) - Single pass
# Space: O(n) - Hash map stores up to n prices

# Example
products = [
    {'name': 'Mouse', 'price': 25},
    {'name': 'Keyboard', 'price': 75},
    {'name': 'Monitor', 'price': 200},
    {'name': 'Webcam', 'price': 100}
]

result = two_sum_products(products, 300)  # Monitor (200) + Webcam (100)
# Returns: (2, 3)
```

💡 **Pro Tip**: If array is sorted, two pointers is more space-efficient (O(1) space).

#### Problem 2: Maximum Subarray (Kadane's Algorithm)

```python
def max_profit_period(daily_profits):
    """
    Find consecutive days with maximum total profit.
    
    Real-world: E-commerce wants to identify best sales period.
    Can have negative profit days (returns/refunds).
    
    Kadane's Algorithm: Keep track of best sum ending at current position.
    """
    if not daily_profits:
        return 0
    
    max_current = max_global = daily_profits[0]
    start = end = 0
    temp_start = 0
    
    for i in range(1, len(daily_profits)):
        # Either extend current subarray or start new one
        if daily_profits[i] > max_current + daily_profits[i]:
            max_current = daily_profits[i]
            temp_start = i
        else:
            max_current = max_current + daily_profits[i]
        
        # Update global maximum
        if max_current > max_global:
            max_global = max_current
            start = temp_start
            end = i
    
    return {
        'max_profit': max_global,
        'start_day': start,
        'end_day': end,
        'period': daily_profits[start:end+1]
    }

# Time: O(n) - Single pass
# Space: O(1) - Only tracking variables

# Example: Week of daily profits (some days had net losses)
profits = [100, -50, 200, -100, 300, 400, -200]
result = max_profit_period(profits)
# max_profit: 750 (days 2-5: 200 + (-100) + 300 + 400)
```

**Memory Trick**: "**K**eep **A**dding **D**aily **A**mounts **N**aturally **E**xcept when starting fresh is better"

#### Problem 3: Product of Array Except Self

```python
def calculate_market_share_excluding_self(company_sales):
    """
    For each company, calculate total market share of all OTHER companies.
    
    Real-world: Show each company what market looks like without them.
    Must do in O(n) time without division (some sales might be 0).
    """
    n = len(company_sales)
    result = [1] * n
    
    # Left pass: result[i] = product of all elements to the left
    left_product = 1
    for i in range(n):
        result[i] = left_product
        left_product *= company_sales[i]
    
    # Right pass: multiply by product of all elements to the right
    right_product = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right_product
        right_product *= company_sales[i]
    
    return result

# Time: O(n) - Two passes
# Space: O(1) - Result array doesn't count as extra space

# Example: 4 companies with sales in millions
sales = [10, 20, 30, 40]  # Company A, B, C, D
market_without_self = calculate_market_share_excluding_self(sales)
# [24000, 12000, 8000, 6000]
# Company A sees market of 20*30*40 = 24,000 (everyone except A)
```

**Visual Diagram:**
```
Array:    [10,  20,  30,  40]
         
Left:     [1,   10,  200, 6000]   (product of elements to left)
Right:    [24000, 1200, 40,  1]    (product of elements to right)
Result:   [24000, 12000, 8000, 6000] (left[i] * right[i])
```

### 2.1.3 Matrix (2D Arrays)

```python
# E-commerce: Product rating matrix (users × products)
def analyze_rating_matrix(ratings):
    """
    ratings[user][product] = rating score
    
    Real-world: Recommendation system analyzing user-product ratings.
    Find average rating per product and most active users.
    """
    if not ratings or not ratings[0]:
        return {}
    
    num_users = len(ratings)
    num_products = len(ratings[0])
    
    # Calculate average rating per product
    product_avg = []
    for product_idx in range(num_products):
        total = sum(ratings[user_idx][product_idx] 
                   for user_idx in range(num_users))
        product_avg.append(total / num_users)
    
    # Find most active user (most ratings > 0)
    user_activity = []
    for user_idx in range(num_users):
        active_count = sum(1 for rating in ratings[user_idx] if rating > 0)
        user_activity.append(active_count)
    
    most_active_user = max(range(num_users), key=lambda i: user_activity[i])
    
    return {
        'product_averages': product_avg,
        'most_active_user': most_active_user,
        'user_activity': user_activity
    }

# Time: O(m * n) where m = users, n = products
# Space: O(n) for storing averages

# Example: 3 users × 4 products
ratings = [
    [5, 4, 0, 5],  # User 0: rated 3 products
    [0, 5, 4, 0],  # User 1: rated 2 products
    [4, 0, 5, 4]   # User 2: rated 3 products
]

result = analyze_rating_matrix(ratings)
# product_averages: [3.0, 3.0, 3.0, 3.0]
# most_active_user: 0 or 2 (both rated 3 products)
```

#### Matrix Traversal Patterns

```python
# Pattern 1: Spiral Order (like warehouse inventory scan)
def scan_warehouse_spiral(warehouse_grid):
    """
    Scan warehouse shelves in spiral pattern.
    
    Real-world: Inventory robot scans shelves spiraling inward.
    """
    if not warehouse_grid:
        return []
    
    result = []
    top, bottom = 0, len(warehouse_grid) - 1
    left, right = 0, len(warehouse_grid[0]) - 1
    
    while top <= bottom and left <= right:
        # Scan top row left to right
        for col in range(left, right + 1):
            result.append(warehouse_grid[top][col])
        top += 1
        
        # Scan right column top to bottom
        for row in range(top, bottom + 1):
            result.append(warehouse_grid[row][right])
        right -= 1
        
        # Scan bottom row right to left
        if top <= bottom:
            for col in range(right, left - 1, -1):
                result.append(warehouse_grid[bottom][col])
            bottom -= 1
        
        # Scan left column bottom to top
        if left <= right:
            for row in range(bottom, top - 1, -1):
                result.append(warehouse_grid[row][left])
            left += 1
    
    return result

# Time: O(m * n)
# Space: O(1) excluding result

# Example: 3×3 warehouse
warehouse = [
    ['A1', 'A2', 'A3'],
    ['B1', 'B2', 'B3'],
    ['C1', 'C2', 'C3']
]
scan_order = scan_warehouse_spiral(warehouse)
# ['A1', 'A2', 'A3', 'B3', 'C3', 'C2', 'C1', 'B1', 'B2']
```

**Visual:**
```
Start → → →
        ↓
    ↑ ← ←
    ↓
    → →
```

---

## 2.2 Strings

### 2.2.1 String Fundamentals

**Definition**: A string is a sequence of characters. In programming, a string can be thought of as an array of characters, but with additional properties and methods specific to text manipulation.

**Key Characteristics:**
- **Sequence**: Ordered collection of characters
- **Immutable (in Python)**: Cannot modify characters in place - must create new string
- **Indexed**: Each character has a position (0-indexed)
- **Unicode support**: Can represent characters from any language

**Key Properties in Python:**
- **Immutable**: Cannot change characters in place
- **Sequence**: Can iterate, slice, index like arrays
- **Unicode**: Supports international characters

```python
# Common string operations
product_name = "MacBook Pro 16-inch"

# Accessing - O(1)
first_char = product_name[0]  # 'M'
last_char = product_name[-1]  # 'h'

# Slicing - O(k) where k = slice length
brand = product_name[:7]  # 'MacBook'
model = product_name[8:11]  # 'Pro'

# Searching - O(n)
index = product_name.find('Pro')  # 8
exists = 'Air' in product_name  # False

# Modification - Creates NEW string (immutability)
upper_name = product_name.upper()  # 'MACBOOK PRO 16-INCH'
replaced = product_name.replace('16', '14')  # 'MacBook Pro 14-inch'

# Splitting - O(n)
words = product_name.split()  # ['MacBook', 'Pro', '16-inch']

# Joining - O(n)
tags = ['laptop', 'apple', 'premium']
tag_string = ', '.join(tags)  # 'laptop, apple, premium'
```

### 2.2.2 String Manipulation Problems

#### Problem 1: Reverse String

```python
# In-place reversal (convert to list first due to immutability)
def reverse_product_code(code):
    """
    Reverse product SKU code for barcode generation.
    
    Real-world: Some barcode systems need reversed codes.
    """
    chars = list(code)
    left, right = 0, len(chars) - 1
    
    while left < right:
        chars[left], chars[right] = chars[right], chars[left]
        left += 1
        right -= 1
    
    return ''.join(chars)

# Time: O(n)
# Space: O(n) - list conversion (unavoidable in Python due to immutability)

# Example
sku = "LAPTOP-2024-001"
reversed_sku = reverse_product_code(sku)  # "100-4202-POTPAL"
```

💡 **Python Shortcut**: `code[::-1]` - but understand the algorithm!

#### Problem 2: Valid Palindrome

```python
def is_palindrome_product_name(name):
    """
    Check if product name reads same forwards and backwards.
    Ignore spaces, punctuation, case.
    
    Real-world: Marketing wants palindrome product names for campaign.
    """
    # Two pointers from both ends
    left, right = 0, len(name) - 1
    
    while left < right:
        # Skip non-alphanumeric from left
        while left < right and not name[left].isalnum():
            left += 1
        
        # Skip non-alphanumeric from right
        while left < right and not name[right].isalnum():
            right -= 1
        
        # Compare (case-insensitive)
        if name[left].lower() != name[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True

# Time: O(n)
# Space: O(1)

# Examples
print(is_palindrome_product_name("A Santa at NASA"))  # True
print(is_palindrome_product_name("race a car"))  # False
print(is_palindrome_product_name("Able was I, ere I saw Elba"))  # True
```

#### Problem 3: Anagram Detection

```python
def are_anagrams(word1, word2):
    """
    Check if two product names are anagrams.
    
    Real-world: Detect similar/copycat product names.
    """
    # Remove spaces and convert to lowercase
    word1 = word1.replace(' ', '').lower()
    word2 = word2.replace(' ', '').lower()
    
    if len(word1) != len(word2):
        return False
    
    # Count character frequencies
    char_count = {}
    
    for char in word1:
        char_count[char] = char_count.get(char, 0) + 1
    
    for char in word2:
        if char not in char_count:
            return False
        char_count[char] -= 1
        if char_count[char] < 0:
            return False
    
    return True

# Time: O(n)
# Space: O(1) - At most 26 letters in English

# Examples
print(are_anagrams("listen", "silent"))  # True
print(are_anagrams("Triangle", "Integral"))  # True
print(are_anagrams("Apple", "Papel"))  # False

# Alternative: Sorting approach
def are_anagrams_v2(word1, word2):
    """Using sorting - simpler but less efficient for very long strings"""
    w1 = ''.join(sorted(word1.replace(' ', '').lower()))
    w2 = ''.join(sorted(word2.replace(' ', '').lower()))
    return w1 == w2

# Time: O(n log n) - Due to sorting
# Space: O(n) - Sorted strings
```

### 2.2.3 Substring Problems

#### Problem 1: Longest Substring Without Repeating Characters

```python
def longest_unique_product_sequence(product_codes):
    """
    Find longest sequence of unique product codes in order history.
    
    Real-world: Analyze customer purchase diversity in single session.
    """
    char_index = {}  # char -> last seen index
    max_length = 0
    start = 0
    
    for end, char in enumerate(product_codes):
        # If char seen and within current window, move start
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        
        char_index[char] = end
        max_length = max(max_length, end - start + 1)
    
    return max_length

# Time: O(n)
# Space: O(min(n, m)) where m = alphabet size

# Example
codes = "ABCABCBB"  # Order history
longest = longest_unique_product_sequence(codes)  # 3 (ABC)

# Another example
codes = "PWWKEW"
longest = longest_unique_product_sequence(codes)  # 3 (WKE or KEW)
```

**Visual Walkthrough:**
```
String: "ABCABCBB"
        
Index:  0 1 2 3 4 5 6 7
Char:   A B C A B C B B

Window [A B C] (0-2): length 3 ✓
Window [A B C A]: A repeats, move start to index 1
Window [B C A] (1-3): length 3 ✓
Window [B C A B]: B repeats, move start to index 2
...

Max length: 3
```

#### Problem 2: Minimum Window Substring

```python
def find_minimum_order_containing_items(order_sequence, required_items):
    """
    Find smallest consecutive order segment containing all required items.
    
    Real-world: Warehouse optimization - minimal pick path containing all items.
    """
    from collections import Counter
    
    if not required_items:
        return ""
    
    required_count = Counter(required_items)
    have_count = {}
    
    required = len(required_count)
    formed = 0
    
    left = 0
    min_length = float('inf')
    min_window = (0, 0)
    
    for right, char in enumerate(order_sequence):
        # Add character to window
        have_count[char] = have_count.get(char, 0) + 1
        
        # Check if frequency matches required
        if char in required_count and have_count[char] == required_count[char]:
            formed += 1
        
        # Try to contract window
        while left <= right and formed == required:
            # Update result if smaller window found
            if right - left + 1 < min_length:
                min_length = right - left + 1
                min_window = (left, right)
            
            # Remove leftmost character
            char = order_sequence[left]
            have_count[char] -= 1
            if char in required_count and have_count[char] < required_count[char]:
                formed -= 1
            
            left += 1
    
    l, r = min_window
    return order_sequence[l:r+1] if min_length != float('inf') else ""

# Time: O(|S| + |T|) where S = order_sequence, T = required_items
# Space: O(|S| + |T|)

# Example
orders = "ADOBECODEBANC"
required = "ABC"
result = find_minimum_order_containing_items(orders, required)
# "BANC" - smallest substring containing A, B, and C
```

### 2.2.4 Pattern Matching

#### Problem: Implement strStr() / Find Needle in Haystack

```python
def find_product_in_catalog(catalog_text, product_name):
    """
    Find first occurrence of product name in catalog.
    
    Real-world: Search product in PDF catalog.
    """
    if not product_name:
        return 0
    
    catalog_len = len(catalog_text)
    pattern_len = len(product_name)
    
    # Try each possible starting position
    for i in range(catalog_len - pattern_len + 1):
        # Check if pattern matches at position i
        match = True
        for j in range(pattern_len):
            if catalog_text[i + j] != product_name[j]:
                match = False
                break
        
        if match:
            return i
    
    return -1

# Time: O(n * m) where n = catalog length, m = pattern length
# Space: O(1)

# Example
catalog = "Our premium laptops include MacBook Pro and Dell XPS"
product = "MacBook"
index = find_product_in_catalog(catalog, product)  # 30

# Python built-in (optimized):
index = catalog.find(product)  # Same result, faster implementation
```

⚠️ **Common Pitfall**: Python's `find()` uses optimized Boyer-Moore-Horspool algorithm - O(n) average case!

---

## Practice Questions - Section 2.1 (Arrays)

### Fill in the Gaps

1. Accessing an element by index in an array is ________ time complexity.
2. The ________ algorithm can find maximum subarray sum in O(n) time.
3. Inserting an element at the beginning of an array requires ________ all other elements.
4. In the "product of array except self" problem, we use two passes: left products and ________ products.
5. Spiral matrix traversal uses four boundaries: top, bottom, left, and ________.

### True or False

1. Arrays in Python can grow dynamically without explicit resizing. **[T/F]**
2. Binary search only works on sorted arrays. **[T/F]**
3. Removing an element from the middle of an array is always O(1). **[T/F]**
4. Two Sum problem can be solved in O(n) time using a hash map. **[T/F]**
5. Matrix traversal always requires O(m*n) space. **[T/F]**

### Multiple Choice

1. What is the time complexity of finding the minimum element in an unsorted array?
   - A) O(1)
   - B) O(log n)
   - C) O(n)
   - D) O(n log n)

2. Which approach is most space-efficient for Two Sum on a sorted array?
   - A) Brute force - O(n²) time, O(1) space
   - B) Hash map - O(n) time, O(n) space
   - C) Two pointers - O(n) time, O(1) space
   - D) Binary search - O(n log n) time, O(1) space

3. Kadane's algorithm is used for:
   - A) Sorting an array
   - B) Finding maximum subarray sum
   - C) Searching in rotated array
   - D) Detecting duplicates

### Code Challenge

```python
def rotate_array(nums, k):
    """
    Rotate array to the right by k steps.
    Example: [1,2,3,4,5], k=2 → [4,5,1,2,3]
    
    Implement in O(n) time and O(1) space.
    """
    # Your code here
    pass
```

---

## Practice Questions - Section 2.2 (Strings)

### Fill in the Gaps

1. Strings in Python are ________, meaning they cannot be modified in place.
2. Checking if one string is an anagram of another using sorting takes ________ time.
3. The sliding window technique for substring problems typically achieves ________ time complexity.
4. In palindrome checking, we typically use ________ pointers from both ends.
5. Python's `find()` method uses the ________ algorithm for pattern matching.

### True or False

1. String concatenation in a loop using += is efficient in Python. **[T/F]**
2. Reversing a string can be done in-place in O(1) space in Python. **[T/F]**
3. Anagram detection requires O(n) space for character frequency counting. **[T/F]**
4. Longest substring without repeating characters requires O(n²) time. **[T/F]**
5. Pattern matching using naive approach has O(n*m) complexity. **[T/F]**

### Multiple Choice

1. What's the most efficient way to check if a string is a palindrome?
   - A) Reverse and compare - O(n) time, O(n) space
   - B) Two pointers - O(n) time, O(1) space
   - C) Recursive approach - O(n) time, O(n) space
   - D) All are equally efficient

2. To find all anagrams in a list of strings, the best approach is:
   - A) Compare each pair - O(n² * m log m)
   - B) Sort all and group - O(n * m log m)
   - C) Use hash map with sorted strings as keys - O(n * m log m)
   - D) Both B and C

3. For minimum window substring problem, which technique is used?
   - A) Two pointers with hash map
   - B) Binary search
   - C) Dynamic programming
   - D) Backtracking

### Code Challenge

```python
def group_anagrams(words):
    """
    Group anagrams together.
    Example: ["eat","tea","tan","ate","nat","bat"]
    Result: [["eat","tea","ate"],["tan","nat"],["bat"]]
    
    Implement efficiently.
    """
    # Your code here
    pass
```

---

## Answers - Section 2.1 (Arrays)

<details>
<summary><strong>View Answers</strong></summary>

### Fill in the Gaps
1. O(1) or constant
2. Kadane's
3. shifting
4. right
5. right

### True or False
1. **True** - Python lists (dynamic arrays) resize automatically
2. **True** - Binary search requires sorted data
3. **False** - O(n) due to shifting remaining elements
4. **True** - Hash map enables O(n) solution
5. **False** - O(1) space excluding output (only tracking variables needed)

### Multiple Choice
1. **C** - Must check all n elements
2. **C** - Two pointers: O(n) time with O(1) space
3. **B** - Kadane's finds maximum subarray sum

### Code Challenge Answer
```python
def rotate_array(nums, k):
    """
    Rotate using reversal technique:
    1. Reverse entire array
    2. Reverse first k elements
    3. Reverse remaining elements
    """
    n = len(nums)
    k = k % n  # Handle k > n
    
    def reverse(start, end):
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1
    
    reverse(0, n - 1)      # Reverse all
    reverse(0, k - 1)      # Reverse first k
    reverse(k, n - 1)      # Reverse rest
    
    return nums

# Time: O(n)
# Space: O(1)

# Example: [1,2,3,4,5], k=2
# After step 1: [5,4,3,2,1]
# After step 2: [4,5,3,2,1]
# After step 3: [4,5,1,2,3] ✓
```

</details>

## Answers - Section 2.2 (Strings)

<details>
<summary><strong>View Answers</strong></summary>

### Fill in the Gaps
1. immutable
2. O(n log n)
3. O(n)
4. two
5. Boyer-Moore-Horspool (or similar optimized algorithm)

### True or False
1. **False** - Creates new string each time, O(n²) total for n concatenations
2. **False** - Must convert to list first (O(n) space) due to immutability
3. **False** - O(1) space for English (max 26 letters), technically O(min(n, alphabet_size))
4. **False** - Can be done in O(n) using sliding window with hash map
5. **True** - Naive approach checks every position with pattern length

### Multiple Choice
1. **B** - Two pointers is optimal: O(n) time, O(1) space
2. **D** - Both sorting and hash map grouping achieve O(n * m log m)
3. **A** - Sliding window with hash map to track character frequencies

### Code Challenge Answer
```python
def group_anagrams(words):
    """Use sorted string as hash key"""
    from collections import defaultdict
    
    anagram_groups = defaultdict(list)
    
    for word in words:
        # Sort characters to create key
        sorted_word = ''.join(sorted(word))
        anagram_groups[sorted_word].append(word)
    
    return list(anagram_groups.values())

# Time: O(n * k log k) where n = number of words, k = max word length
# Space: O(n * k) for storing results

# Example
words = ["eat","tea","tan","ate","nat","bat"]
result = group_anagrams(words)
# [["eat","tea","ate"], ["tan","nat"], ["bat"]]
```

</details>

---

## LeetCode Problems - Arrays & Strings

### Arrays - Easy
- ✅ 1. Two Sum
- ✅ 26. Remove Duplicates from Sorted Array
- ✅ 27. Remove Element
- ✅ 88. Merge Sorted Array
- ✅ 121. Best Time to Buy and Sell Stock
- ✅ 217. Contains Duplicate
- ✅ 242. Valid Anagram
- ✅ 283. Move Zeroes

### Arrays - Medium
- 🟨 3. Longest Substring Without Repeating Characters
- 🟨 11. Container With Most Water
- 🟨 15. 3Sum
- 🟨 33. Search in Rotated Sorted Array
- 🟨 34. Find First and Last Position of Element
- 🟨 48. Rotate Image
- 🟨 53. Maximum Subarray (Kadane's)
- 🟨 54. Spiral Matrix
- 🟨 56. Merge Intervals
- 🟨 57. Insert Interval
- 🟨 75. Sort Colors (Dutch National Flag)
- 🟨 128. Longest Consecutive Sequence
- 🟨 152. Maximum Product Subarray
- 🟨 238. Product of Array Except Self
- 🟨 289. Game of Life

### Arrays - Hard
- 🔴 4. Median of Two Sorted Arrays
- 🔴 42. Trapping Rain Water
- 🔴 84. Largest Rectangle in Histogram

### Strings - Easy
- ✅ 20. Valid Parentheses
- ✅ 125. Valid Palindrome
- ✅ 242. Valid Anagram
- ✅ 344. Reverse String
- ✅ 387. First Unique Character in a String

### Strings - Medium
- 🟨 3. Longest Substring Without Repeating Characters
- 🟨 5. Longest Palindromic Substring
- 🟨 49. Group Anagrams
- 🟨 76. Minimum Window Substring
- 🟨 271. Encode and Decode Strings
- 🟨 424. Longest Repeating Character Replacement

### Strings - Hard
- 🔴 30. Substring with Concatenation of All Words
- 🔴 76. Minimum Window Substring

---

*Continue to: [3. Linked Lists →](03-linked-lists.md)*