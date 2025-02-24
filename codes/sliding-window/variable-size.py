# Find the length of the longest subarray,
# with the same value in each position.
def longestSubarray(nums):
    length = 0
    l = 0

    for r in range(len(nums)):
        if nums[l] != nums[r]:
            l = r
        length = max(length, r - l + 1)

    return length

print(longestSubarray([4,2,2,3,3,3]))
print(longestSubarray([4,2,2,2,2,3,3,3]))

# Find the minimum length subarray,
# where the sum is greater than or equal to the target.
# Assume all values are positive.
def minLength(nums, target):
    length = len(nums)
    l, total = 0, 0

    for r in range(len(nums)):
        total += nums[r]
        while total >= target:
            length = min(length, r - l + 1)
            total -= nums[l]
            l += 1

    return 0 if length == len(nums) else length

print(minLength([2,3,1,2,4,3], 6))