# O(n * k) time complexity where k is the number of bits (i.e. log₂(max_value))
# Best for integers, not for negative numbers unless adjusted.

# How it works:
# Radix sort processes the bits of the number from least significant bit (LSB) to most significant bit (MSB).
# For each bit position, it partitions the array into two groups:
# Numbers with that bit = 0
# Numbers with that bit = 1
# This is done using bitwise AND: (num >> bit) & 1

def radixSort(nums):
    def get_max_bits(max_num):
        bits = 0
        while (max_num >> bits):
            bits += 1
        return bits
        
    max_num = max(nums)
    # max_bits = max_num.bit_length()
    max_bits = get_max_bits(max_num)
    
    for bit in range(max_bits):
        zero_bucket = []
        one_bucket = []
        for n in nums:
            if (n >> bit) & 1:
                one_bucket.append(n)
            else:
                zero_bucket.append(n)
        nums = zero_bucket + one_bucket
    return nums

nums = [2, 3, 5, 4]
print(radixSort(nums))