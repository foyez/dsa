# 1. Fundamentals

## 1.1 Time & Space Complexity

### 1.1.1 Big O Notation

**Definition**: Big O notation is a mathematical notation that describes the limiting behavior of a function when the argument tends towards a particular value or infinity. In computer science, it's used to classify algorithms according to how their running time or space requirements grow as the input size grows.

**Formal Definition**: We say that f(n) = O(g(n)) if there exist positive constants c and n₀ such that:
```
0 ≤ f(n) ≤ c·g(n) for all n ≥ n₀
```

In simpler terms: Big O describes the **upper bound** or worst-case scenario of how an algorithm's performance scales with input size. It tells us the maximum amount of time or space an algorithm will need.

**Why Big O?** It allows us to:
- Compare algorithm efficiency independent of hardware
- Predict how algorithms scale to large inputs
- Make informed decisions about which algorithm to use
- Communicate algorithm efficiency concisely

**Real-World Analogy**: Think of Big O like shipping methods for an e-commerce platform:
- **O(1)**: Same-day drone delivery - always takes the same time regardless of order size
- **O(log n)**: Regional hub distribution - divide regions in half each time
- **O(n)**: Processing each item on a conveyor belt - time grows linearly with items
- **O(n log n)**: Sorting packages by zip code before delivery
- **O(n²)**: Comparing every customer review with every other review for duplicates
- **O(2ⁿ)**: Trying every possible combination of gift wrapping options

### Common Time Complexities (Best to Worst)

| Complexity | Name | Example Operation | Growth Rate |
|------------|------|-------------------|-------------|
| O(1) | Constant | Array access by index | 1 |
| O(log n) | Logarithmic | Binary search | Very slow |
| O(n) | Linear | Linear search | Moderate |
| O(n log n) | Linearithmic | Merge sort, Quick sort | Fast |
| O(n²) | Quadratic | Bubble sort, nested loops | Fast |
| O(n³) | Cubic | 3 nested loops | Very fast |
| O(2ⁿ) | Exponential | Fibonacci (naive) | Explosive |
| O(n!) | Factorial | Permutations | Catastrophic |

**Memory Trick - "LC-NQQEF"**:
- **L**og, **C**onstant, **N**linear, **Q**uadratic, **Q**ubic, **E**xponential, **F**actorial

### Visual Comparison

```
Time (operations) for n = 100:

O(1)       : ▪                    (1 operation)
O(log n)   : ▪▪▪▪▪▪▪             (7 operations)
O(n)       : ▪▪▪▪▪▪▪▪▪▪...       (100 operations)
O(n log n) : ▪▪▪▪▪▪▪▪▪▪▪▪...     (664 operations)
O(n²)      : ▪▪▪▪▪▪▪▪▪▪▪▪▪▪...   (10,000 operations)
O(2ⁿ)      : ████████████████... (1.26 × 10³⁰ operations - impossible!)
```

### Real-World Examples

#### O(1) - Constant Time
```python
# E-commerce: Get product price by ID
def get_product_price(product_catalog, product_id):
    """
    Access product price directly from hash map.
    Like looking up a product on a shelf by its exact location.
    """
    return product_catalog[product_id]['price']

# Time: O(1) - Direct hash map access
# Space: O(1) - No extra space used

# Example usage
catalog = {
    'LAPTOP-001': {'name': 'MacBook Pro', 'price': 1999},
    'PHONE-001': {'name': 'iPhone 15', 'price': 999}
}
price = get_product_price(catalog, 'LAPTOP-001')  # Always same time
```

#### O(log n) - Logarithmic Time
```python
# E-commerce: Find product in sorted price list
def find_product_in_price_range(sorted_products, target_price):
    """
    Binary search through sorted products.
    Like using a phone book - always check the middle and eliminate half.
    """
    left, right = 0, len(sorted_products) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if sorted_products[mid]['price'] == target_price:
            return sorted_products[mid]
        elif sorted_products[mid]['price'] < target_price:
            left = mid + 1
        else:
            right = mid - 1
    
    return None

# Time: O(log n) - Halve search space each iteration
# Space: O(1) - Only a few variables

# Example: Finding $999 phone in 1000 products takes ~10 comparisons
```

#### O(n) - Linear Time
```python
# E-commerce: Calculate total cart value
def calculate_cart_total(shopping_cart):
    """
    Sum all item prices in cart.
    Must visit each item once - like scanning items at checkout.
    """
    total = 0
    for item in shopping_cart:
        total += item['price'] * item['quantity']
    
    return total

# Time: O(n) - Process each item once
# Space: O(1) - Only total variable

# Example usage
cart = [
    {'name': 'Laptop', 'price': 1999, 'quantity': 1},
    {'name': 'Mouse', 'price': 29, 'quantity': 2},
    {'name': 'Keyboard', 'price': 89, 'quantity': 1}
]
total = calculate_cart_total(cart)  # 2206
```

#### O(n log n) - Linearithmic Time
```python
# E-commerce: Sort products by rating and price
def sort_products_by_rating_price(products):
    """
    Efficient sorting using merge sort or Python's Timsort.
    Like organizing inventory by multiple criteria.
    """
    return sorted(products, key=lambda p: (-p['rating'], p['price']))

# Time: O(n log n) - Optimal comparison-based sorting
# Space: O(n) - Timsort uses auxiliary space

# Example usage
products = [
    {'name': 'Laptop A', 'rating': 4.5, 'price': 1999},
    {'name': 'Laptop B', 'rating': 4.5, 'price': 1799},
    {'name': 'Laptop C', 'rating': 4.8, 'price': 2199}
]
sorted_products = sort_products_by_rating_price(products)
# Result: Laptop C (4.8), Laptop B (4.5, cheaper), Laptop A (4.5)
```

#### O(n²) - Quadratic Time
```python
# E-commerce: Find duplicate reviews (naive approach)
def find_duplicate_reviews_naive(reviews):
    """
    Compare every review with every other review.
    Like checking each customer against all others - inefficient!
    """
    duplicates = []
    n = len(reviews)
    
    for i in range(n):
        for j in range(i + 1, n):
            if reviews[i]['text'] == reviews[j]['text']:
                duplicates.append((reviews[i], reviews[j]))
    
    return duplicates

# Time: O(n²) - Nested loops, n(n-1)/2 comparisons
# Space: O(k) - k is number of duplicates found

# Better approach: Use hash set - O(n) time
def find_duplicate_reviews_optimized(reviews):
    """Using hash set for O(n) solution"""
    seen = set()
    duplicates = []
    
    for review in reviews:
        if review['text'] in seen:
            duplicates.append(review)
        else:
            seen.add(review['text'])
    
    return duplicates

# Time: O(n) - Single pass with hash lookups
# Space: O(n) - Store all unique review texts
```

#### O(2ⁿ) - Exponential Time
```python
# E-commerce: Generate all possible product bundles (naive)
def generate_all_bundles(products):
    """
    Generate every possible combination of products.
    Like trying every possible gift basket combination - explodes quickly!
    """
    def backtrack(index, current_bundle):
        if index == len(products):
            result.append(current_bundle[:])
            return
        
        # Include current product
        current_bundle.append(products[index])
        backtrack(index + 1, current_bundle)
        current_bundle.pop()
        
        # Exclude current product
        backtrack(index + 1, current_bundle)
    
    result = []
    backtrack(0, [])
    return result

# Time: O(2ⁿ) - Each product: include or exclude = 2 choices
# Space: O(n) - Recursion depth
# For 20 products: 1,048,576 combinations!
# For 30 products: 1,073,741,824 combinations - impractical!
```

### 1.1.2 Space Complexity

**Definition**: Space complexity is the total amount of memory space used by an algorithm as a function of the input size. It includes both:
1. **Auxiliary space**: Extra space used by the algorithm (temporary variables, recursion stack)
2. **Input space**: Space used by the input itself

**What counts towards space complexity:**
- Variables and constants
- Recursion call stack
- Dynamically allocated memory (arrays, hash maps, objects)
- Data structures created during execution

**What typically doesn't count:**
- Input data itself (unless we modify or copy it)
- Code space (the compiled program)

**Space complexity** measures memory used relative to input size.

```python
# Social Network: Different space complexities

# O(1) - Constant space
def count_mutual_friends(user1_friends, user2_friends):
    """
    Count without storing results.
    Like counting on fingers without writing down.
    """
    count = 0
    for friend in user1_friends:
        if friend in user2_friends:
            count += 1
    return count
# Space: O(1) - Only counter variable

# O(n) - Linear space
def get_mutual_friends(user1_friends, user2_friends):
    """
    Store all mutual friends.
    Like making a list of common connections.
    """
    return list(set(user1_friends) & set(user2_friends))
# Space: O(n) - Store results in new list

# O(n²) - Quadratic space
def build_friend_matrix(users, friendships):
    """
    Create adjacency matrix for social network.
    Like having a table showing all possible connections.
    """
    n = len(users)
    matrix = [[0] * n for _ in range(n)]
    
    for user1, user2 in friendships:
        matrix[user1][user2] = 1
        matrix[user2][user1] = 1
    
    return matrix
# Space: O(n²) - n×n matrix
# For 10,000 users: 100,000,000 cells!
```

### 1.1.3 Amortized Analysis

**Definition**: Amortized analysis is a method of analyzing algorithms that considers the average performance of each operation over a sequence of operations, rather than analyzing each operation in isolation. It provides a more accurate picture when some operations are expensive but rare.

**Why Amortized Analysis?** 
- Some operations may be costly (O(n)) but occur infrequently
- Most operations are cheap (O(1))
- Average cost per operation over many operations is what matters
- More realistic performance prediction than worst-case for individual operations

**Three methods of amortized analysis:**
1. **Aggregate method**: Total cost of n operations / n
2. **Accounting method**: Assign different charges to operations
3. **Potential method**: Use a potential function to represent stored work

**Amortized analysis** considers average cost over a sequence of operations.

```python
# E-commerce: Dynamic array for shopping cart
class ShoppingCart:
    """
    Shopping cart with dynamic resizing.
    Like a shopping basket that gets bigger when full.
    """
    def __init__(self):
        self.capacity = 4
        self.size = 0
        self.items = [None] * self.capacity
    
    def add_item(self, product):
        """
        Add item to cart, resize if needed.
        
        Individual adds:
        - Usually O(1) - just append
        - Occasionally O(n) - when resizing needed
        
        Amortized: O(1) - resize cost spread across all operations
        """
        # Resize if full (happens rarely)
        if self.size == self.capacity:
            self._resize()
        
        self.items[self.size] = product
        self.size += 1
    
    def _resize(self):
        """Double capacity when full"""
        self.capacity *= 2
        new_items = [None] * self.capacity
        
        # Copy old items - O(n) operation
        for i in range(self.size):
            new_items[i] = self.items[i]
        
        self.items = new_items

# Time analysis for adding n items:
# - Most operations: O(1)
# - Resize at capacities: 4, 8, 16, 32, ..., n
# - Total resize cost: n/2 + n/4 + n/8 + ... ≈ n
# - Amortized per operation: O(n) / n = O(1)

cart = ShoppingCart()
for i in range(100):
    cart.add_item(f"Product {i}")  # Each add is O(1) amortized
```

### Key Rules for Analyzing Complexity

**Rule 1: Drop Constants**
```python
# Both are O(n), not O(2n) or O(n + 10)
def process_orders_v1(orders):
    for order in orders:  # O(n)
        validate(order)
    for order in orders:  # O(n)
        ship(order)
    # Total: O(2n) → O(n)

def process_orders_v2(orders):
    for order in orders:  # O(n)
        validate(order)
        ship(order)
    # Total: O(n)
```

**Rule 2: Drop Non-Dominant Terms**
```python
# O(n² + n) → O(n²)
# O(n² + log n) → O(n²)
# O(2ⁿ + n²) → O(2ⁿ)

def find_similar_products(products, reviews):
    # Compare products: O(n²)
    for i in range(len(products)):
        for j in range(i + 1, len(products)):
            compare(products[i], products[j])
    
    # Process reviews: O(m)
    for review in reviews:
        process(review)
    
    # Total: O(n² + m)
    # If m < n, then O(n²)
    # If m and n independent, keep both
```

**Rule 3: Different Variables for Different Inputs**
```python
# Wrong: O(n)
# Correct: O(n + m) where n = products, m = reviews

def cross_reference_products_reviews(products, reviews):
    """Process two independent input arrays"""
    # Process all products
    for product in products:  # O(n)
        index_product(product)
    
    # Process all reviews
    for review in reviews:  # O(m)
        index_review(review)
    
    # Time: O(n + m) - NOT O(n)!
```

---

## 1.2 Problem-Solving Patterns

### 1.2.1 Two Pointers Pattern

**Definition**: The two pointers technique uses two pointers (indices or references) that traverse a data structure in a coordinated way to solve problems efficiently. The pointers can move in the same direction, opposite directions, or at different speeds.

**When to Use Two Pointers:**
- Working with sorted arrays or linked lists
- Finding pairs, triplets, or subarrays with specific properties
- Removing duplicates in-place
- Comparing elements from both ends
- Partitioning arrays

**Time Complexity**: Usually O(n) instead of O(n²) brute force  
**Space Complexity**: O(1) - only using pointers

**Types of Two Pointer Patterns:**
1. **Opposite Direction** (left and right pointers meet in middle)
2. **Same Direction** (slow and fast pointers, both move forward)
3. **Different Speeds** (fast pointer moves faster than slow)

---

#### Pattern 1: Opposite Direction (Converging Pointers)

**Template:**
```python
def two_pointers_opposite(arr):
    """
    Template for opposite direction two pointers.
    Use when: Finding pairs with target sum, palindrome checking,
              reversing, partitioning around pivot
    """
    left = 0
    right = len(arr) - 1
    
    while left < right:
        # Process current pair
        # Make decision based on comparison
        
        if condition_met:
            # Process and/or return result
            return result
        elif need_larger_value:
            left += 1    # Move left pointer right
        else:
            right -= 1   # Move right pointer left
    
    return default_result

# Time: O(n) - each element visited at most once
# Space: O(1) - only pointer variables
```

**Problem: Two Sum II (Sorted Array)**
```python
def two_sum_sorted(numbers, target):
    """
    LeetCode 167: Two Sum II - Input Array Is Sorted
    
    Given sorted array, find two numbers that add up to target.
    Return indices (1-indexed).
    
    Why two pointers works:
    - Array is sorted
    - If sum too small, need larger number → move left right
    - If sum too large, need smaller number → move right left
    - Guaranteed to find answer if it exists
    """
    left = 0
    right = len(numbers) - 1
    
    while left < right:
        current_sum = numbers[left] + numbers[right]
        
        if current_sum == target:
            return [left + 1, right + 1]  # 1-indexed
        elif current_sum < target:
            left += 1   # Need larger sum
        else:
            right -= 1  # Need smaller sum
    
    return []  # No solution found

# Time: O(n)
# Space: O(1)

# Example
nums = [2, 7, 11, 15]
target = 9
print(two_sum_sorted(nums, target))  # [1, 2]

# Visual walkthrough:
# [2, 7, 11, 15], target = 9
#  L           R   → 2 + 15 = 17 > 9, move R left
# [2, 7, 11, 15]
#  L       R       → 2 + 11 = 13 > 9, move R left  
# [2, 7, 11, 15]
#  L   R           → 2 + 7 = 9 ✓ Found!
```

**Problem: Three Sum**
```python
def three_sum(nums):
    """
    LeetCode 15: 3Sum
    
    Find all unique triplets that sum to zero.
    Pattern: Fix one number, use two pointers for remaining two.
    
    Key insight: Sort first, then fix first number and use
    two pointers on remaining array.
    """
    nums.sort()  # O(n log n)
    result = []
    
    for i in range(len(nums) - 2):
        # Skip duplicates for first number
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        # Two pointers for remaining array
        left = i + 1
        right = len(nums) - 1
        target = -nums[i]  # Want nums[left] + nums[right] = -nums[i]
        
        while left < right:
            current_sum = nums[left] + nums[right]
            
            if current_sum == target:
                result.append([nums[i], nums[left], nums[right]])
                
                # Skip duplicates for second number
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                # Skip duplicates for third number
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                
                left += 1
                right -= 1
            elif current_sum < target:
                left += 1
            else:
                right -= 1
    
    return result

# Time: O(n²) - O(n log n) sort + O(n²) for nested loops
# Space: O(1) or O(n) depending on sort implementation

# Example
nums = [-1, 0, 1, 2, -1, -4]
print(three_sum(nums))  # [[-1, -1, 2], [-1, 0, 1]]
```

**Problem: Container With Most Water**
```python
def max_area(height):
    """
    LeetCode 11: Container With Most Water
    
    Find two lines that form container with most water.
    
    Greedy approach: Start with widest container, move pointer
    of shorter line inward (moving taller line won't increase area)
    """
    left = 0
    right = len(height) - 1
    max_water = 0
    
    while left < right:
        # Calculate current area
        width = right - left
        current_height = min(height[left], height[right])
        current_area = width * current_height
        max_water = max(max_water, current_area)
        
        # Move pointer of shorter line
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_water

# Time: O(n)
# Space: O(1)

# Example
heights = [1, 8, 6, 2, 5, 4, 8, 3, 7]
print(max_area(heights))  # 49 (indices 1 and 8: 8 * min(8,7) = 49)
```

---

#### Pattern 2: Same Direction (Fast & Slow Pointers)

**Template:**
```python
def two_pointers_same_direction(arr):
    """
    Template for same direction two pointers.
    Use when: Removing duplicates, partitioning, in-place modifications
    
    Slow pointer: Write position
    Fast pointer: Read position
    """
    slow = 0  # Write pointer
    
    for fast in range(len(arr)):
        # Process element at fast pointer
        
        if should_keep_element(arr[fast]):
            # Keep this element
            arr[slow] = arr[fast]
            slow += 1
    
    return slow  # New length / write position

# Time: O(n)
# Space: O(1)
```

**Problem: Remove Duplicates from Sorted Array**
```python
def remove_duplicates(nums):
    """
    LeetCode 26: Remove Duplicates from Sorted Array
    
    Remove duplicates in-place, return new length.
    Array is sorted, so duplicates are consecutive.
    
    Slow pointer: Position to write next unique element
    Fast pointer: Scan through array
    """
    if not nums:
        return 0
    
    slow = 1  # First element always unique
    
    for fast in range(1, len(nums)):
        # If current element different from previous
        if nums[fast] != nums[fast - 1]:
            nums[slow] = nums[fast]
            slow += 1
    
    return slow  # New length

# Time: O(n)
# Space: O(1)

# Example
nums = [1, 1, 2, 2, 2, 3, 4, 4]
length = remove_duplicates(nums)
print(nums[:length])  # [1, 2, 3, 4]

# Visual:
# [1, 1, 2, 2, 2, 3, 4, 4]
#  s  f  → nums[f] == nums[f-1], skip
# [1, 1, 2, 2, 2, 3, 4, 4]
#  s     f  → nums[f] != nums[f-1], write and move slow
# [1, 2, 2, 2, 2, 3, 4, 4]
#     s     f  → Continue pattern...
```

**Problem: Move Zeroes**
```python
def move_zeroes(nums):
    """
    LeetCode 283: Move Zeroes
    
    Move all zeros to end while maintaining relative order.
    
    Slow pointer: Position to write next non-zero
    Fast pointer: Scan for non-zeros
    """
    slow = 0  # Write position for non-zeros
    
    # Move all non-zeros to front
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow] = nums[fast]
            slow += 1
    
    # Fill remaining with zeros
    for i in range(slow, len(nums)):
        nums[i] = 0

# Time: O(n)
# Space: O(1)

# Example
nums = [0, 1, 0, 3, 12]
move_zeroes(nums)
print(nums)  # [1, 3, 12, 0, 0]
```

---

#### Pattern 3: Different Speeds (Cycle Detection)

**Template:**
```python
def fast_slow_pointers(head):
    """
    Template for fast/slow pointers (Floyd's algorithm).
    Use when: Detecting cycles, finding middle, finding Kth from end
    
    Slow: Moves 1 step at a time
    Fast: Moves 2 steps at a time
    """
    if not head or not head.next:
        return None
    
    slow = head
    fast = head
    
    while fast and fast.next:
        slow = slow.next         # 1 step
        fast = fast.next.next    # 2 steps
        
        if slow == fast:
            # Cycle detected or other condition met
            return True
    
    return False

# Time: O(n)
# Space: O(1)
```

**Problem: Linked List Cycle**
```python
def has_cycle(head):
    """
    LeetCode 141: Linked List Cycle
    
    Detect if linked list has a cycle.
    
    Why it works:
    - If no cycle, fast reaches end
    - If cycle exists, fast eventually catches slow
    - Like runners on circular track
    """
    if not head or not head.next:
        return False
    
    slow = head
    fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            return True  # Cycle detected
    
    return False  # No cycle

# Time: O(n)
# Space: O(1)
```

---

#### Two Pointers: Decision Tree

```
Is array/list sorted?
│
├─ YES → Consider opposite direction pointers
│         ├─ Finding pair with target sum → Two Sum II pattern
│         ├─ Finding triplets → 3Sum pattern (fix one + two pointers)
│         └─ Palindrome check → Compare from both ends
│
└─ NO → Consider same direction pointers
          ├─ Remove duplicates → Slow/fast pattern
          ├─ Partition array → Dutch National Flag
          ├─ Linked list cycle → Fast/slow pattern
          └─ Find middle element → Fast/slow pattern
```

---

## Two Pointers - LeetCode Problems (NeetCode.io)

### Easy ✅
- **167. Two Sum II - Input Array Is Sorted** - Classic opposite direction
- **125. Valid Palindrome** - Opposite direction with filtering
- **283. Move Zeroes** - Same direction, in-place modification
- **26. Remove Duplicates from Sorted Array** - Same direction
- **27. Remove Element** - Same direction

### Medium 🟨
- **15. 3Sum** - Fix one + two pointers (IMPORTANT)
- **11. Container With Most Water** - Greedy with two pointers
- **16. 3Sum Closest** - Variation of 3Sum
- **18. 4Sum** - Extension of 3Sum pattern
- **75. Sort Colors** - Dutch National Flag (3 pointers)
- **80. Remove Duplicates from Sorted Array II** - Same direction variant
- **151. Reverse Words in a String** - Multiple two pointer passes
- **186. Reverse Words in a String II** - In-place reversal
- **259. 3Sum Smaller** - Count triplets variant
- **287. Find the Duplicate Number** - Fast/slow cycle detection
- **344. Reverse String** - Simple opposite direction
- **345. Reverse Vowels of a String** - Opposite with filtering
- **360. Sort Transformed Array** - Two pointers on parabola
- **457. Circular Array Loop** - Cycle detection variant
- **524. Longest Word in Dictionary through Deleting** - Two pointers comparison
- **713. Subarray Product Less Than K** - Sliding window variant
- **826. Most Profit Assigning Work** - Greedy two pointers
- **844. Backspace String Compare** - Two pointers from end
- **881. Boats to Save People** - Greedy pairing
- **925. Long Pressed Name** - String matching with pointers
- **986. Interval List Intersections** - Merge intervals pattern
- **1750. Minimum Length of String After Deleting Similar Ends** - Opposite direction

### Hard 🔴
- **42. Trapping Rain Water** - Two pointers tracking max heights
- **76. Minimum Window Substring** - Sliding window (covered in next section)

---

## Two Pointers - Common Patterns Summary

| Pattern | Use Case | Example | Complexity |
|---------|----------|---------|------------|
| **Opposite Direction** | Sorted array pairs/triplets | Two Sum II, 3Sum | O(n) or O(n²) |
| **Same Direction** | In-place array modification | Remove Duplicates | O(n) |
| **Fast/Slow** | Cycle detection, middle finding | Linked List Cycle | O(n) |
| **Three Pointers** | Partitioning, 4Sum | Sort Colors | O(n) |
| **Multiple Passes** | Complex transformations | Reverse Words | O(n) |

**Real-World Analogy**: Like two workers on an assembly line - one starts from each end, meeting in the middle; or one slow worker doing quality checks while a fast worker scans ahead.

```python
def find_products_for_budget(sorted_prices, budget):
    """
    Two pointers to find pair of products within budget.
    Products sorted by price.
    
    Like: Customer has $500, find 2 items that total exactly $500.
    """
    left = 0
    right = len(sorted_prices) - 1
    
    while left < right:
        current_sum = sorted_prices[left]['price'] + sorted_prices[right]['price']
        
        if current_sum == budget:
            return (sorted_prices[left], sorted_prices[right])
        elif current_sum < budget:
            left += 1  # Need higher price, move left pointer right
        else:
            right -= 1  # Need lower price, move right pointer left
    
    return None

# Time: O(n) - Single pass with two pointers
# Space: O(1) - Only pointer variables

# Example
products = [
    {'name': 'Mouse', 'price': 25},
    {'name': 'Keyboard', 'price': 75},
    {'name': 'Monitor', 'price': 200},
    {'name': 'Laptop', 'price': 700}
]
result = find_products_for_budget(products, 275)  # Mouse + Monitor
```

**Common Two Pointer Patterns:**

1. **Opposite Direction** (above example)
2. **Same Direction** (fast & slow)
3. **Sliding Window** (next section)

```python
# Same direction: Remove duplicates from customer order
def remove_duplicate_items(order_items):
    """
    Remove consecutive duplicate items from order.
    
    Like: Order shows [Laptop, Laptop, Mouse, Keyboard, Keyboard]
          Result: [Laptop, Mouse, Keyboard]
    """
    if not order_items:
        return 0
    
    write_pos = 1  # Position to write unique item
    
    for read_pos in range(1, len(order_items)):
        if order_items[read_pos] != order_items[read_pos - 1]:
            order_items[write_pos] = order_items[read_pos]
            write_pos += 1
    
    return write_pos  # Length of unique items

# Time: O(n)
# Space: O(1) - In-place modification

# Example
order = ['Laptop', 'Laptop', 'Mouse', 'Keyboard', 'Keyboard']
unique_len = remove_duplicate_items(order)
print(order[:unique_len])  # ['Laptop', 'Mouse', 'Keyboard']
```

### 1.2.2 Sliding Window Pattern

🎯 **Use When**: Finding subarrays/substrings with specific properties

**Real-World**: Like viewing products through a shop window - slide to see different sets.

```python
# E-commerce: Maximum revenue in any k-day period
def max_revenue_k_days(daily_revenue, k):
    """
    Find maximum revenue in any consecutive k days.
    
    Like: Black Friday analysis - which 3-day period had highest sales?
    """
    if len(daily_revenue) < k:
        return 0
    
    # Calculate first window
    window_revenue = sum(daily_revenue[:k])
    max_revenue = window_revenue
    
    # Slide window: remove leftmost, add rightmost
    for i in range(k, len(daily_revenue)):
        window_revenue = window_revenue - daily_revenue[i - k] + daily_revenue[i]
        max_revenue = max(max_revenue, window_revenue)
    
    return max_revenue

# Time: O(n) - Single pass after initial window
# Space: O(1) - Only tracking window sum

# Example: 7 days of revenue, find best 3-day period
revenue = [2000, 3000, 1000, 4000, 5000, 2000, 3000]
best = max_revenue_k_days(revenue, 3)  # 11000 (days 4-6: 4000+5000+2000)
```

**Variable Window Size:**

```python
# E-commerce: Smallest order size with minimum total value
def smallest_order_with_min_value(item_values, min_total):
    """
    Find smallest number of consecutive items with total >= min_total.
    
    Like: Bulk order discount - minimum 10 items worth $500+
    """
    if not item_values:
        return 0
    
    left = 0
    current_sum = 0
    min_length = float('inf')
    
    for right in range(len(item_values)):
        current_sum += item_values[right]
        
        # Shrink window while sum meets requirement
        while current_sum >= min_total:
            min_length = min(min_length, right - left + 1)
            current_sum -= item_values[left]
            left += 1
    
    return min_length if min_length != float('inf') else 0

# Time: O(n) - Each element visited at most twice
# Space: O(1)

# Example
values = [10, 20, 30, 15, 25, 40, 35]
result = smallest_order_with_min_value(values, 80)  # 2 items (40+35=75 not enough, 25+40=65, 15+25+40=80)
```

### 1.2.3 Fast & Slow Pointers

🎯 **Use When**: Detecting cycles, finding middle element, or linked list problems

**Real-World**: Like two runners on a track - one runs twice as fast.

```python
# Social Network: Detect circular reference in user recommendations
class UserNode:
    def __init__(self, user_id, name):
        self.user_id = user_id
        self.name = name
        self.recommended_by = None  # Next node in chain

def has_circular_recommendations(start_user):
    """
    Detect if recommendation chain loops back.
    
    Like: User A recommended by B, B by C, C by A (circular!)
    
    Floyd's Cycle Detection (Tortoise and Hare)
    """
    if not start_user or not start_user.recommended_by:
        return False
    
    slow = start_user
    fast = start_user
    
    while fast and fast.recommended_by:
        slow = slow.recommended_by           # Move 1 step
        fast = fast.recommended_by.recommended_by  # Move 2 steps
        
        if slow == fast:  # They meet = cycle exists!
            return True
    
    return False

# Time: O(n) - Fast catches slow in at most n iterations
# Space: O(1) - Only two pointers

# Example usage
alice = UserNode(1, "Alice")
bob = UserNode(2, "Bob")
charlie = UserNode(3, "Charlie")

alice.recommended_by = bob
bob.recommended_by = charlie
charlie.recommended_by = alice  # Creates cycle!

print(has_circular_recommendations(alice))  # True
```

**Finding Middle Element:**

```python
# E-commerce: Find median review score efficiently
class ReviewNode:
    def __init__(self, score, next_review=None):
        self.score = score
        self.next = next_review

def find_median_review(head):
    """
    Find middle review in linked list without knowing length.
    
    Like: Finding median customer satisfaction without counting all reviews first.
    """
    if not head:
        return None
    
    slow = fast = head
    
    # When fast reaches end, slow is at middle
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow.score

# Time: O(n)
# Space: O(1)

# Example: Reviews linked list with scores: 5 → 4 → 3 → 5 → 2
review1 = ReviewNode(5)
review2 = ReviewNode(4)
review3 = ReviewNode(3)  # Middle
review4 = ReviewNode(5)
review5 = ReviewNode(2)

review1.next = review2
review2.next = review3
review3.next = review4
review4.next = review5

median = find_median_review(review1)  # 3
```

### 1.2.4 Pattern Recognition Guide

| Pattern | When to Use | Time | Space | Key Indicators |
|---------|-------------|------|-------|----------------|
| Two Pointers | Sorted array, pairs/triplets | O(n) | O(1) | "Find pair", "sorted array" |
| Sliding Window | Subarray/substring problems | O(n) | O(1) | "Consecutive elements", "subarray" |
| Fast & Slow | Cycle detection, middle element | O(n) | O(1) | "Linked list", "cycle", "middle" |
| Hash Map | Frequency count, lookup | O(n) | O(n) | "Count", "frequency", "duplicate" |
| Binary Search | Sorted array search | O(log n) | O(1) | "Sorted", "search" |
| BFS | Shortest path, level-order | O(V+E) | O(V) | "Shortest", "level", "minimum moves" |
| DFS | Paths, cycles, connectivity | O(V+E) | O(h) | "All paths", "connected", "reachable" |
| Backtracking | Combinations, permutations | O(2ⁿ/n!) | O(n) | "All combinations", "generate all" |
| DP | Optimization, count ways | Varies | O(n²) | "Maximum/minimum", "count ways" |
| Greedy | Optimal substructure | O(n log n) | O(1) | "Maximize/minimize", intervals |

---

## Practice Questions - Section 1.1

### Fill in the Gaps

1. The time complexity O(n log n) is typically associated with efficient ________ algorithms like merge sort.
2. When we say an algorithm runs in O(1) time, it means the runtime is ________ regardless of input size.
3. The space complexity of recursive algorithms is often determined by the maximum ________ depth.
4. Dropping constants means O(2n + 5) simplifies to ________.
5. Amortized analysis considers the ________ cost of operations over a sequence.

### True or False

1. O(n²) is always slower than O(n log n) for all values of n. **[T/F]**
2. An algorithm with O(1) space complexity uses no memory at all. **[T/F]**
3. Binary search has O(log n) time complexity because it halves the search space each iteration. **[T/F]**
4. O(n + m) and O(n) are equivalent when n and m represent different inputs. **[T/F]**
5. Recursive functions always have at least O(n) space complexity. **[T/F]**

### Multiple Choice

1. Which operation on a hash map typically runs in O(1) time?
   - A) Iterating through all keys
   - B) Finding the maximum value
   - C) Looking up a value by key
   - D) Sorting all values

2. What is the time complexity of accessing an element in an array by index?
   - A) O(log n)
   - B) O(1)
   - C) O(n)
   - D) O(n²)

3. Which has better time complexity for large inputs?
   - A) O(log n)
   - B) O(n)
   - C) O(n²)
   - D) All are equivalent

4. An algorithm does 3n² + 2n + 1 operations. What is its Big O complexity?
   - A) O(1)
   - B) O(n)
   - C) O(n²)
   - D) O(3n² + 2n)

### Code Challenge

```python
# Analyze the time and space complexity
def mystery_function(products, target_category):
    result = []
    categories = {}
    
    # Step 1
    for product in products:
        if product['category'] not in categories:
            categories[product['category']] = []
        categories[product['category']].append(product)
    
    # Step 2
    if target_category in categories:
        sorted_products = sorted(categories[target_category], 
                                key=lambda x: x['price'])
        result = sorted_products[:10]
    
    return result
```

**Question**: What are the time and space complexities?
- Assume n = total products, k = products in target category

---

## Practice Questions - Section 1.2

### Fill in the Gaps

1. The two pointers pattern works best on ________ arrays when finding pairs.
2. Sliding window is ideal for finding the maximum sum of ________ k elements.
3. Fast and slow pointers can detect a ________ in a linked list.
4. The sliding window technique avoids recalculating by ________ one element and adding another.
5. Two pointers running in opposite directions typically achieve ________ time complexity.

### True or False

1. Sliding window can only be used with fixed-size windows. **[T/F]**
2. Fast and slow pointers require O(n) extra space to detect cycles. **[T/F]**
3. Two pointers pattern always requires the array to be sorted first. **[T/F]**
4. Variable-size sliding window can solve problems in better than O(n²) time. **[T/F]**
5. The fast pointer in cycle detection always moves exactly twice as fast as the slow pointer. **[T/F]**

### Multiple Choice

1. Which pattern is best for finding the longest substring without repeating characters?
   - A) Two Pointers (opposite direction)
   - B) Sliding Window (variable size)
   - C) Fast & Slow Pointers
   - D) Binary Search

2. To find a pair with a given sum in a sorted array, which is most efficient?
   - A) Nested loops - O(n²)
   - B) Two pointers - O(n)
   - C) Hash map - O(n)
   - D) Both B and C are equally good

3. What does it indicate when fast and slow pointers meet in a linked list?
   - A) The list is sorted
   - B) A cycle exists
   - C) The list length is even
   - D) The middle element is found

### Code Challenge

```python
# Fix this implementation
def find_max_sum_subarray(arr, k):
    """Find maximum sum of k consecutive elements"""
    max_sum = 0
    for i in range(len(arr) - k + 1):
        current_sum = 0
        for j in range(i, i + k):
            current_sum += arr[j]
        max_sum = max(max_sum, current_sum)
    return max_sum
```

**Question**: This implementation is O(n*k). Rewrite using sliding window to achieve O(n).

---

## Answers

### Section 1.1 Answers

### Fill in the Gaps
1. sorting
2. constant
3. recursion (or call stack)
4. O(n)
5. average

### True or False
1. **False** - For very small n (like n=1), they're equal. We mean for large n.
2. **False** - O(1) means constant space, not zero space (still uses variables).
3. **True** - Binary search eliminates half the elements each step.
4. **False** - They are different; we must keep both variables.
5. **False** - Iterative recursion or tail recursion can be O(1) space.

### Multiple Choice
1. **C** - Hash map lookup is O(1) average case
2. **B** - Array index access is always O(1)
3. **A** - O(log n) grows slowest
4. **C** - Drop constants and lower terms = O(n²)

### Code Challenge Answer
```
Time Complexity: O(n + k log k)
- Step 1 (grouping): O(n) - iterate all products
- Step 2 (sorting): O(k log k) - sort k products in target category
- Overall: O(n + k log k)

Space Complexity: O(n)
- categories dictionary: O(n) - stores all products
- result list: O(k) - at most 10 products
- Overall: O(n)
```

### Section 1.2 Answers

### Fill in the Gaps
1. sorted
2. consecutive (or any k)
3. cycle
4. removing (or subtracting)
5. O(n) or linear

### True or False
1. **False** - Variable-size windows are common (e.g., smallest subarray with sum ≥ k)
2. **False** - Only requires O(1) space (two pointers)
3. **False** - Can be used on unsorted arrays (e.g., remove duplicates in-place)
4. **True** - Reduces nested loop O(n²) to single pass O(n)
5. **True** - Typically fast moves 2 steps, slow moves 1 step

### Multiple Choice
1. **B** - Variable sliding window expands/contracts to find longest valid substring
2. **D** - Both are O(n) and optimal; two pointers uses O(1) space, hash map uses O(n) space
3. **B** - Meeting point indicates cycle exists (Floyd's algorithm)

**Code Challenge - Optimized Solution:**
```python
def find_max_sum_subarray(arr, k):
    """Sliding window approach - O(n)"""
    if len(arr) < k:
        return 0
    
    # Calculate first window
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # Slide window
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

# Time: O(n) - single pass after initial window
# Space: O(1)
```

---

## Interview Questions - Fundamentals

### Question 1: Explain Big O notation to a non-technical person

**Sample Answer:**
"Big O notation is like describing how long a task takes as the work increases. Imagine you're a teacher grading papers:

- **O(1)**: Looking at the top paper - same time whether you have 10 or 1000 papers
- **O(n)**: Grading each paper one by one - doubles time when papers double  
- **O(n²)**: Comparing every paper with every other paper for plagiarism - explodes quickly
- **O(log n)**: Like finding a name in a phone book by repeatedly opening to the middle - very efficient even with millions of entries

We care about Big O because it tells us how well our solution will handle large amounts of data, which is critical in real applications."

### Question 2: When would you choose O(n) space to achieve O(n) time vs O(1) space with O(n²) time?

**Sample Answer:**
"This is a classic time-space tradeoff. I'd consider:

**Choose O(n) space for O(n) time when:**
- User experience depends on speed (e.g., search results, checkout process)
- Data size is manageable (millions of records, not billions)
- Memory is available and cheaper than compute time
- Example: Caching product catalog in hash map for instant lookups

**Choose O(1) space with O(n²) time when:**
- Memory is extremely constrained (embedded systems, mobile devices)
- Data is small enough that O(n²) is acceptable
- Data is being streamed and can't be stored
- Example: Finding duplicates in a small settings file during device startup

In modern web applications, I typically optimize for time because user experience is paramount and memory is relatively cheap. However, in big data scenarios with billions of records, memory becomes the bottleneck and we need space-efficient solutions."

### Question 3: How do you decide between different patterns for a problem?

**Sample Answer:**
"I follow a systematic approach:

1. **Identify constraints**: Sorted? Linked list? Size limits?
2. **Pattern matching**: 
   - 'Pairs/triplets' + sorted → Two pointers
   - 'Consecutive elements' → Sliding window
   - 'Cycle' + linked list → Fast & slow pointers
   - 'Count/frequency' → Hash map
   
3. **Analyze complexity**: Can I do better than brute force?

Example: 'Find two products with prices summing to $500'
- Brute force: O(n²) - check all pairs
- Hash map: O(n) - store prices, check if (500 - price) exists
- Two pointers (if sorted): O(n) with O(1) space

I'd choose two pointers if data is already sorted, otherwise hash map for unsorted data, depending on whether the array is large enough that sorting overhead O(n log n) matters."

### Question 4: Explain a time when you optimized code complexity

**Sample Answer (Prepare your own):**
"At [Company], we had a report generation feature that was timing out for large accounts. The original implementation used nested loops to match transactions with invoices - O(n*m) where n=transactions, m=invoices.

I optimized it by:
1. Building a hash map of invoices by ID - O(m) time, O(m) space
2. Single pass through transactions looking up invoices - O(n) time
3. Total: O(n + m) instead of O(n*m)

For accounts with 10,000 transactions and 5,000 invoices:
- Before: 50,000,000 operations (~30 seconds)
- After: 15,000 operations (~0.1 seconds)

This shows that analyzing complexity isn't just academic - it directly impacts user experience."

---

*Continue to: [2. Arrays & Strings →](02-arrays-strings.md)*