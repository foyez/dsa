# 12. Bit Manipulation

## 12.1 Bitwise Operations Fundamentals

### 12.1.1 Binary Number System

**Definition**: Binary is a base-2 number system using only 0 and 1. Each position represents a power of 2.

**Binary Representation:**
```
Decimal 13 = Binary 1101

Position:  3  2  1  0
Power:     2³ 2² 2¹ 2⁰
Binary:    1  1  0  1
Value:     8  4  0  1  = 13

Formula: 1×2³ + 1×2² + 0×2¹ + 1×2⁰ = 8 + 4 + 0 + 1 = 13
```

**Signed Integers (Two's Complement):**
- Positive numbers: Standard binary
- Negative numbers: Two's complement
- Most significant bit (MSB) indicates sign (0 = positive, 1 = negative)

```python
# Python integers are arbitrary precision (no fixed size)
# But conceptually, for 8-bit signed integers:

5 in binary (8-bit):   00000101
-5 in two's complement:
  Step 1: Invert bits  11111010
  Step 2: Add 1        11111011

# Verify: 
# 11111011 = -128 + 64 + 32 + 16 + 8 + 2 + 1 = -5
```

**Bit Positions:**
```
Number: 1 0 1 1 0 1 0 0
Bit:    7 6 5 4 3 2 1 0  (0-indexed from right)
        ^               ^
        MSB             LSB
```

---

### 12.1.2 Basic Bitwise Operators

#### AND (&)

**Definition**: Returns 1 if both bits are 1, otherwise 0.

```python
a = 12  # 1100
b = 10  # 1010
result = a & b  # 1000 = 8

# Truth table:
# 0 & 0 = 0
# 0 & 1 = 0
# 1 & 0 = 0
# 1 & 1 = 1

# Applications:
# 1. Check if bit is set
num = 13  # 1101
mask = 1 << 2  # 0100 (bit 2)
is_set = (num & mask) != 0  # True

# 2. Clear specific bits
num = 15  # 1111
mask = ~(1 << 2)  # 1011 (clear bit 2)
result = num & mask  # 1011 = 11

# 3. Extract lower bits
num = 27  # 11011
lower_4_bits = num & 0xF  # 1011 = 11 (extract lower 4 bits)
```

---

#### OR (|)

**Definition**: Returns 1 if at least one bit is 1, otherwise 0.

```python
a = 12  # 1100
b = 10  # 1010
result = a | b  # 1110 = 14

# Truth table:
# 0 | 0 = 0
# 0 | 1 = 1
# 1 | 0 = 1
# 1 | 1 = 1

# Applications:
# 1. Set specific bit
num = 8  # 1000
mask = 1 << 2  # 0100 (bit 2)
result = num | mask  # 1100 = 12

# 2. Combine flags
READ = 1 << 0   # 0001
WRITE = 1 << 1  # 0010
EXEC = 1 << 2   # 0100
permissions = READ | WRITE  # 0011
```

---

#### XOR (^)

**Definition**: Returns 1 if bits are different, 0 if same.

```python
a = 12  # 1100
b = 10  # 1010
result = a ^ b  # 0110 = 6

# Truth table:
# 0 ^ 0 = 0
# 0 ^ 1 = 1
# 1 ^ 0 = 1
# 1 ^ 1 = 0

# Important properties:
# 1. x ^ 0 = x (identity)
# 2. x ^ x = 0 (self-inverse)
# 3. x ^ y ^ y = x (cancellation)
# 4. Commutative: x ^ y = y ^ x
# 5. Associative: (x ^ y) ^ z = x ^ (y ^ z)

# Applications:
# 1. Toggle bit
num = 12  # 1100
mask = 1 << 2  # 0100
result = num ^ mask  # 1000 = 8

# 2. Swap without temp variable
a, b = 5, 7
a = a ^ b
b = a ^ b  # b = (a ^ b) ^ b = a
a = a ^ b  # a = (a ^ b) ^ a = b
# Now a = 7, b = 5

# 3. Find unique element (all others appear twice)
nums = [2, 3, 2, 4, 3]
unique = 0
for num in nums:
    unique ^= num  # unique = 4
```

---

#### NOT (~)

**Definition**: Inverts all bits (0 becomes 1, 1 becomes 0).

```python
a = 12  # 0000 1100 (assuming 8-bit)
result = ~a  # 1111 0011 = -13 (two's complement)

# In Python, ~x = -x - 1 (due to arbitrary precision)
print(~12)  # -13
print(~(-5))  # 4

# Application:
# Create mask for clearing bits
mask = ~(1 << 3)  # Clear bit 3
# If 1 << 3 = 0000 1000
# Then ~(1 << 3) = 1111 0111
```

---

#### Left Shift (<<)

**Definition**: Shifts bits to the left, fills right with 0s. Equivalent to multiplying by 2^n.

```python
a = 5  # 0101
result = a << 2  # 010100 = 20

# Formula: x << n = x × 2^n
print(5 << 1)  # 10 (5 × 2)
print(5 << 2)  # 20 (5 × 4)
print(5 << 3)  # 40 (5 × 8)

# Applications:
# 1. Fast multiplication by powers of 2
num = 7
doubled = num << 1  # 14

# 2. Create bit masks
mask = 1 << 5  # 100000 (bit 5 set)

# 3. Set specific bit
num = 10  # 1010
num |= (1 << 2)  # Set bit 2: 1110 = 14
```

---

#### Right Shift (>>)

**Definition**: Shifts bits to the right, discards rightmost bits. Equivalent to dividing by 2^n.

```python
a = 20  # 10100
result = a >> 2  # 00101 = 5

# Formula: x >> n = x ÷ 2^n (integer division)
print(20 >> 1)  # 10 (20 ÷ 2)
print(20 >> 2)  # 5 (20 ÷ 4)
print(21 >> 1)  # 10 (21 ÷ 2, rounded down)

# Two types:
# 1. Logical shift: Fill left with 0s
# 2. Arithmetic shift: Fill left with sign bit (preserves sign)

# In Python, >> is arithmetic shift
print(-8 >> 1)  # -4 (sign preserved)

# Applications:
# 1. Fast division by powers of 2
num = 100
halved = num >> 1  # 50

# 2. Extract specific bits
num = 0b11010110
middle_bits = (num >> 3) & 0b111  # Extract bits 3-5
```

---

### 12.1.3 Common Bit Manipulation Tricks

#### Check if Number is Power of 2

```python
def is_power_of_two(n):
    """
    Power of 2 has exactly one bit set.
    Example: 8 = 1000, 8-1 = 0111
    8 & 7 = 0000 = 0
    """
    return n > 0 and (n & (n - 1)) == 0

# Examples
print(is_power_of_two(8))   # True (1000)
print(is_power_of_two(6))   # False (0110)
print(is_power_of_two(16))  # True (10000)
```

---

#### Count Set Bits (Hamming Weight)

```python
def count_set_bits(n):
    """
    LeetCode 191: Number of 1 Bits
    
    Count number of 1s in binary representation.
    """
    count = 0
    while n:
        count += n & 1  # Check LSB
        n >>= 1  # Shift right
    return count

# Brian Kernighan's algorithm (faster):
def count_set_bits_fast(n):
    """
    n & (n-1) removes rightmost set bit.
    """
    count = 0
    while n:
        n &= (n - 1)  # Clear rightmost set bit
        count += 1
    return count

# Examples
print(count_set_bits(11))  # 3 (1011)
print(count_set_bits(128))  # 1 (10000000)

# Python built-in:
print(bin(11).count('1'))  # 3
```

---

#### Get/Set/Clear/Toggle Bit

```python
def get_bit(num, i):
    """Get bit at position i"""
    return (num >> i) & 1

def set_bit(num, i):
    """Set bit at position i to 1"""
    return num | (1 << i)

def clear_bit(num, i):
    """Clear bit at position i to 0"""
    return num & ~(1 << i)

def toggle_bit(num, i):
    """Toggle bit at position i"""
    return num ^ (1 << i)

# Examples
num = 12  # 1100

print(get_bit(num, 2))     # 1
print(set_bit(num, 0))     # 1101 = 13
print(clear_bit(num, 2))   # 1000 = 8
print(toggle_bit(num, 1))  # 1110 = 14
```

---

#### Swap Two Numbers

```python
def swap_xor(a, b):
    """Swap without temporary variable using XOR"""
    a = a ^ b
    b = a ^ b  # b = (a ^ b) ^ b = a
    a = a ^ b  # a = (a ^ b) ^ a = b
    return a, b

# Or using tuple unpacking (Pythonic way):
def swap_pythonic(a, b):
    a, b = b, a
    return a, b

# Examples
print(swap_xor(5, 7))  # (7, 5)
```

---

#### Get Rightmost Set Bit

```python
def rightmost_set_bit(n):
    """
    Get position of rightmost set bit.
    
    n & -n isolates rightmost set bit.
    Example: n = 12 (1100), -n = -12 (0100 in two's complement)
    12 & -12 = 0100 = 4
    """
    return n & -n

# Examples
print(rightmost_set_bit(12))  # 4 (0100)
print(rightmost_set_bit(10))  # 2 (0010)
print(rightmost_set_bit(7))   # 1 (0001)
```

---

#### Clear Rightmost Set Bit

```python
def clear_rightmost_bit(n):
    """
    Clear rightmost set bit.
    
    n & (n-1) clears rightmost set bit.
    Example: n = 12 (1100), n-1 = 11 (1011)
    12 & 11 = 1000 = 8
    """
    return n & (n - 1)

# Examples
print(clear_rightmost_bit(12))  # 8 (1000)
print(clear_rightmost_bit(10))  # 8 (1000)
print(clear_rightmost_bit(7))   # 6 (0110)
```

---

## 12.2 Common Bit Manipulation Problems

### 12.2.1 Single Number

```python
def single_number(nums):
    """
    LeetCode 136: Single Number
    
    Every element appears twice except one. Find the single one.
    
    Approach: XOR all numbers.
    - x ^ x = 0
    - x ^ 0 = x
    - All duplicates cancel out
    """
    result = 0
    for num in nums:
        result ^= num
    return result

# Time: O(n)
# Space: O(1)

# Example
print(single_number([2, 2, 1]))  # 1
print(single_number([4, 1, 2, 1, 2]))  # 4
```

---

### 12.2.2 Single Number II

```python
def single_number_ii(nums):
    """
    LeetCode 137: Single Number II
    
    Every element appears three times except one. Find the single one.
    
    Approach: Count bits at each position.
    If count % 3 != 0, that bit belongs to single number.
    """
    result = 0
    
    for i in range(32):
        bit_sum = 0
        
        # Count how many numbers have bit i set
        for num in nums:
            bit_sum += (num >> i) & 1
        
        # If not divisible by 3, single number has this bit
        if bit_sum % 3:
            result |= (1 << i)
    
    # Handle negative numbers (Python specific)
    if result >= 2**31:
        result -= 2**32
    
    return result

# Time: O(32n) = O(n)
# Space: O(1)

# Example
print(single_number_ii([2, 2, 3, 2]))  # 3
print(single_number_ii([0, 1, 0, 1, 0, 1, 99]))  # 99
```

---

### 12.2.3 Single Number III

```python
def single_number_iii(nums):
    """
    LeetCode 260: Single Number III
    
    Every element appears twice except two. Find both.
    
    Approach:
    1. XOR all numbers → gets x ^ y (two unique numbers)
    2. Find any set bit in x ^ y (indicates difference)
    3. Partition numbers by that bit
    4. XOR each partition separately
    """
    # Get x ^ y
    xor = 0
    for num in nums:
        xor ^= num
    
    # Get rightmost set bit (where x and y differ)
    rightmost_bit = xor & -xor
    
    # Partition and XOR
    x = y = 0
    for num in nums:
        if num & rightmost_bit:
            x ^= num
        else:
            y ^= num
    
    return [x, y]

# Time: O(n)
# Space: O(1)

# Example
print(single_number_iii([1, 2, 1, 3, 2, 5]))  # [3, 5]
```

---

### 12.2.4 Reverse Bits

```python
def reverse_bits(n):
    """
    LeetCode 190: Reverse Bits
    
    Reverse bits of 32-bit unsigned integer.
    """
    result = 0
    
    for i in range(32):
        # Get LSB of n
        bit = n & 1
        
        # Add to result at position 31-i
        result |= (bit << (31 - i))
        
        # Shift n right
        n >>= 1
    
    return result

# Time: O(1) - fixed 32 iterations
# Space: O(1)

# Alternative: bit manipulation trick
def reverse_bits_optimized(n):
    """Divide and conquer approach"""
    # Swap pairs
    n = ((n & 0xAAAAAAAA) >> 1) | ((n & 0x55555555) << 1)
    # Swap nibbles
    n = ((n & 0xCCCCCCCC) >> 2) | ((n & 0x33333333) << 2)
    # Swap bytes
    n = ((n & 0xF0F0F0F0) >> 4) | ((n & 0x0F0F0F0F) << 4)
    # Swap 2-byte pairs
    n = ((n & 0xFF00FF00) >> 8) | ((n & 0x00FF00FF) << 8)
    # Swap 4-byte pairs
    n = (n >> 16) | ((n & 0xFFFF) << 16)
    
    return n

# Example
print(bin(reverse_bits(0b00000010100101000001111010011100)))
# Result: 964176192 (reversed)
```

---

### 12.2.5 Hamming Distance

```python
def hamming_distance(x, y):
    """
    LeetCode 461: Hamming Distance
    
    Count positions where bits are different.
    
    Approach: XOR gives positions where different, count 1s.
    """
    xor = x ^ y
    
    # Count set bits
    count = 0
    while xor:
        xor &= (xor - 1)
        count += 1
    
    return count

# Time: O(1) - at most 32 bits
# Space: O(1)

# Example
print(hamming_distance(1, 4))  # 2
# 1 = 0001
# 4 = 0100
# XOR = 0101 → 2 different bits
```

---

### 12.2.6 Power of Four

```python
def is_power_of_four(n):
    """
    LeetCode 342: Power of Four
    
    Check if n is power of 4.
    
    Approach:
    1. Must be power of 2 (one bit set)
    2. Bit must be at even position (0, 2, 4, ...)
    """
    # Check power of 2
    if n <= 0 or (n & (n - 1)) != 0:
        return False
    
    # Check if bit at even position
    # 0x55555555 = 01010101... (bits at even positions)
    return (n & 0x55555555) != 0

# Alternative: math approach
def is_power_of_four_v2(n):
    """Power of 4 ⟺ power of 2 AND (n-1) divisible by 3"""
    return n > 0 and (n & (n - 1)) == 0 and (n - 1) % 3 == 0

# Examples
print(is_power_of_four(16))  # True (10000)
print(is_power_of_four(5))   # False
print(is_power_of_four(1))   # True (1 = 4^0)
```

---

## 12.3 Advanced Bit Manipulation

### 12.3.1 Subset Generation

```python
def subsets_bitmask(nums):
    """
    Generate all subsets using bitmask.
    
    For n elements, there are 2^n subsets.
    Each number from 0 to 2^n-1 represents a subset.
    """
    n = len(nums)
    result = []
    
    # Iterate through all possible masks
    for mask in range(1 << n):  # 2^n subsets
        subset = []
        
        # Check each bit
        for i in range(n):
            if mask & (1 << i):
                subset.append(nums[i])
        
        result.append(subset)
    
    return result

# Time: O(n × 2^n)
# Space: O(2^n)

# Example
print(subsets_bitmask([1, 2, 3]))
# [[], [1], [2], [1,2], [3], [1,3], [2,3], [1,2,3]]
```

---

### 12.3.2 Maximum XOR of Two Numbers

```python
def find_maximum_xor(nums):
    """
    LeetCode 421: Maximum XOR of Two Numbers in an Array
    
    Find maximum XOR of any two numbers.
    
    Approach: Build answer bit by bit from MSB to LSB.
    Try to set each bit if possible.
    """
    max_xor = 0
    mask = 0
    
    # Process from MSB to LSB (assume 32-bit)
    for i in range(31, -1, -1):
        mask |= (1 << i)
        
        # Get prefixes of all numbers up to bit i
        prefixes = {num & mask for num in nums}
        
        # Try to set bit i in result
        temp = max_xor | (1 << i)
        
        # Check if any two prefixes can give this result
        for prefix in prefixes:
            if temp ^ prefix in prefixes:
                max_xor = temp
                break
    
    return max_xor

# Time: O(32n) = O(n)
# Space: O(n)

# Example
print(find_maximum_xor([3, 10, 5, 25, 2, 8]))  # 28
# 5 ^ 25 = 0101 ^ 11001 = 11100 = 28
```

---

### 12.3.3 Bitwise AND of Numbers Range

```python
def range_bitwise_and(left, right):
    """
    LeetCode 201: Bitwise AND of Numbers Range
    
    Find AND of all numbers in range [left, right].
    
    Key insight: Find common prefix of left and right.
    All bits that change in range will become 0.
    """
    shift = 0
    
    # Find common prefix
    while left < right:
        left >>= 1
        right >>= 1
        shift += 1
    
    # Shift back to original position
    return left << shift

# Time: O(log n)
# Space: O(1)

# Example
print(range_bitwise_and(5, 7))  # 4
# 5 = 101
# 6 = 110
# 7 = 111
# AND = 100 = 4

print(range_bitwise_and(0, 0))  # 0
print(range_bitwise_and(1, 2147483647))  # 0
```

---

### 12.3.4 Sum of Two Integers (No +/- operators)

```python
def get_sum(a, b):
    """
    LeetCode 371: Sum of Two Integers
    
    Add two integers without using + or -.
    
    Approach:
    - XOR gives sum without carry
    - AND gives carry positions
    - Shift carry left and repeat
    """
    mask = 0xFFFFFFFF  # 32-bit mask for Python
    
    while b != 0:
        # Sum without carry
        sum_without_carry = (a ^ b) & mask
        
        # Carry
        carry = ((a & b) << 1) & mask
        
        a = sum_without_carry
        b = carry
    
    # Handle negative (Python specific)
    return a if a <= 0x7FFFFFFF else ~(a ^ mask)

# Time: O(1) - at most 32 iterations
# Space: O(1)

# Example
print(get_sum(1, 2))   # 3
print(get_sum(-1, 1))  # 0
```

---

## 12.4 Bit Manipulation Patterns

### 12.4.1 Check Even/Odd

```python
def is_even(n):
    """Check if number is even"""
    return (n & 1) == 0

def is_odd(n):
    """Check if number is odd"""
    return (n & 1) == 1

# Examples
print(is_even(4))  # True
print(is_odd(7))   # True
```

---

### 12.4.2 Multiply/Divide by 2^n

```python
def multiply_by_power_of_2(num, n):
    """Multiply by 2^n"""
    return num << n

def divide_by_power_of_2(num, n):
    """Divide by 2^n"""
    return num >> n

# Examples
print(multiply_by_power_of_2(5, 3))  # 40 (5 × 8)
print(divide_by_power_of_2(40, 3))   # 5 (40 ÷ 8)
```

---

### 12.4.3 Find Missing Number

```python
def missing_number(nums):
    """
    LeetCode 268: Missing Number
    
    Array contains n distinct numbers in range [0, n].
    Find missing number.
    
    Approach: XOR all indices and values.
    Missing number won't cancel out.
    """
    result = len(nums)
    
    for i, num in enumerate(nums):
        result ^= i ^ num
    
    return result

# Time: O(n)
# Space: O(1)

# Example
print(missing_number([3, 0, 1]))  # 2
print(missing_number([0, 1]))     # 2
```

---

### 12.4.4 UTF-8 Validation

```python
def valid_utf8(data):
    """
    LeetCode 393: UTF-8 Validation
    
    Check if array represents valid UTF-8 encoding.
    
    UTF-8 rules:
    1-byte: 0xxxxxxx
    2-byte: 110xxxxx 10xxxxxx
    3-byte: 1110xxxx 10xxxxxx 10xxxxxx
    4-byte: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
    """
    n_bytes = 0
    
    for num in data:
        # Get relevant 8 bits
        byte = num & 0xFF
        
        if n_bytes == 0:
            # Determine number of bytes
            if (byte >> 7) == 0:
                n_bytes = 0
            elif (byte >> 5) == 0b110:
                n_bytes = 1
            elif (byte >> 4) == 0b1110:
                n_bytes = 2
            elif (byte >> 3) == 0b11110:
                n_bytes = 3
            else:
                return False
        else:
            # Must be continuation byte (10xxxxxx)
            if (byte >> 6) != 0b10:
                return False
            n_bytes -= 1
    
    return n_bytes == 0

# Time: O(n)
# Space: O(1)

# Example
print(valid_utf8([197, 130, 1]))  # True
print(valid_utf8([235, 140, 4]))  # False
```

---

## 12.5 Bit Manipulation in Interviews

### 12.5.1 Common Patterns

**Pattern 1: XOR for Uniqueness**
- Finding unique elements when others appear even times
- XOR properties: x ^ x = 0, x ^ 0 = x

**Pattern 2: Bit Counting**
- Count set bits (Hamming weight)
- Use n & (n-1) to clear rightmost bit

**Pattern 3: Bit Masking**
- Set/clear/toggle specific bits
- Check if bit is set

**Pattern 4: Powers of 2**
- Check: n & (n-1) == 0
- Related: powers of 4, alignment checks

**Pattern 5: Subset Generation**
- Use bitmask to represent subsets
- Iterate 0 to 2^n - 1

---

### 12.5.2 Optimization Techniques

```python
# 1. Fast modulo with power of 2
def fast_modulo(n, divisor):
    """n % divisor when divisor is power of 2"""
    return n & (divisor - 1)

print(fast_modulo(17, 8))  # 17 % 8 = 1
# 17 & 7 = 10001 & 00111 = 00001 = 1


# 2. Check if two numbers have opposite signs
def opposite_signs(x, y):
    """Check if x and y have opposite signs"""
    return (x ^ y) < 0

print(opposite_signs(5, -3))   # True
print(opposite_signs(-5, -3))  # False


# 3. Conditional set value (no branching)
def conditional_set(condition, val1, val2):
    """Return val1 if condition else val2"""
    return val1 if condition else val2
    # Bit hack version:
    # return (val1 * condition) + (val2 * (not condition))


# 4. Absolute value
def absolute(n):
    """Get absolute value using bit manipulation"""
    mask = n >> 31  # All 1s if negative, all 0s if positive
    return (n + mask) ^ mask

print(absolute(-5))  # 5
print(absolute(5))   # 5


# 5. Min/Max without branching
def min_bitwise(a, b):
    """Get minimum without if statement"""
    return b ^ ((a ^ b) & -(a < b))

def max_bitwise(a, b):
    """Get maximum without if statement"""
    return a ^ ((a ^ b) & -(a < b))
```

---

## Practice Questions

### Fill in the Gaps

1. The XOR of a number with itself equals ________.
2. Left shift by n positions multiplies by ________.
3. To check if a number is power of 2, use ________.
4. The operation n & (n-1) removes the ________ set bit.
5. To isolate the rightmost set bit, use ________.

### True or False

1. x ^ 0 = x for any x. **[T/F]**
2. Left shift is always faster than multiplication. **[T/F]**
3. Bitwise AND can be used to check if a bit is set. **[T/F]**
4. XOR is commutative and associative. **[T/F]**
5. Right shift always fills with zeros. **[T/F]**

### Multiple Choice

1. What does n & (n-1) do?
   - A) Sets rightmost bit
   - B) Clears rightmost bit
   - C) Toggles all bits
   - D) Counts bits

2. Time complexity of counting set bits?
   - A) O(1)
   - B) O(log n)
   - C) O(number of set bits)
   - D) O(n)

3. To check if bit i is set, use:
   - A) n | (1 << i)
   - B) n & (1 << i)
   - C) n ^ (1 << i)
   - D) n >> i

### Code Challenge

```python
def count_bits(n):
    """
    LeetCode 338: Counting Bits
    
    Return array where ans[i] is number of 1s in binary of i.
    
    Example: n = 5
    Output: [0, 1, 1, 2, 1, 2]
    
    0 → 0 (0 ones)
    1 → 1 (1 one)
    2 → 10 (1 one)
    3 → 11 (2 ones)
    4 → 100 (1 one)
    5 → 101 (2 ones)
    
    Solve in O(n) using DP and bit manipulation.
    Hint: ans[i] = ans[i >> 1] + (i & 1)
    """
    # Your code here
    pass
```

---

## Answers

<details>
<summary><strong>View Answers</strong></summary>

### Fill in the Gaps

1. **0** (x ^ x = 0)
2. **2^n**
3. **n & (n-1) == 0** (and n > 0)
4. **rightmost** (or least significant)
5. **n & -n** (or n & (~n + 1))

### True or False

1. **True** - XOR with 0 is identity operation
2. **False** - Modern compilers optimize equally; shift not always faster
3. **True** - (n & (1 << i)) checks if bit i is set
4. **True** - Both properties hold for XOR
5. **False** - Arithmetic shift fills with sign bit; logical shift with 0

### Multiple Choice

1. **B** - Clears the rightmost set bit
2. **C** - Depends on number of set bits (with Brian Kernighan)
3. **B** - AND with mask checks if bit is set

### Code Challenge Answer

```python
def count_bits(n):
    """
    DP approach using bit manipulation.
    
    Observation: ans[i] = ans[i >> 1] + (i & 1)
    - i >> 1 removes rightmost bit
    - i & 1 checks if rightmost bit is 1
    """
    ans = [0] * (n + 1)
    
    for i in range(1, n + 1):
        # Number of 1s in i = number of 1s in (i/2) + (i%2)
        ans[i] = ans[i >> 1] + (i & 1)
    
    return ans

# Time: O(n)
# Space: O(n)

# Alternative: Using Brian Kernighan
def count_bits_v2(n):
    ans = [0] * (n + 1)
    
    for i in range(1, n + 1):
        # ans[i] = ans[i & (i-1)] + 1
        # Removing rightmost bit gives previous state
        ans[i] = ans[i & (i - 1)] + 1
    
    return ans

# Example
print(count_bits(5))  # [0, 1, 1, 2, 1, 2]
```

</details>

---

## LeetCode Problems (NeetCode.io)

### Bit Manipulation - Easy ✅
- 136. Single Number (IMPORTANT)
- 191. Number of 1 Bits
- 268. Missing Number
- 338. Counting Bits
- 389. Find the Difference
- 461. Hamming Distance

### Bit Manipulation - Medium 🟨
- 7. Reverse Integer
- 29. Divide Two Integers
- 137. Single Number II (IMPORTANT)
- 190. Reverse Bits
- 201. Bitwise AND of Numbers Range
- 260. Single Number III
- 371. Sum of Two Integers
- 393. UTF-8 Validation

### Bit Manipulation - Hard 🔴
- 421. Maximum XOR of Two Numbers in an Array

---

## Summary

### Quick Reference

**Basic Operations:**
- AND (&): Check/clear bits
- OR (|): Set bits
- XOR (^): Toggle bits, find unique
- NOT (~): Invert bits
- << : Multiply by 2^n
- \>> : Divide by 2^n

**Common Tricks:**
- Power of 2: `n & (n-1) == 0`
- Set bit i: `n | (1 << i)`
- Clear bit i: `n & ~(1 << i)`
- Toggle bit i: `n ^ (1 << i)`
- Get bit i: `(n >> i) & 1`
- Rightmost bit: `n & -n`
- Clear rightmost: `n & (n-1)`

**Patterns:**
- XOR for uniqueness
- Bitmask for subsets
- Count bits for analysis
- Bit shifts for optimization

---

*Continue to: [13. Advanced Topics →](13-advanced-topics.md)*