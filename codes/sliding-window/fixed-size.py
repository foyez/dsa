# Fixed size sliding window
# Write a function called maxSubarraySum which accepts
# an array of integers and a number called n.
# The function should calculate the maximum sum of n consecutive elements in the array.
def maxSubSum(arr, n):
    if not arr:
        return None

    window_sum = max_window_sum = 0
    l = 0

    for r, val in enumerate(arr):
        if r - l + 1 > n:
            window_sum = window_sum - arr[l] + val
            l += 1
        else:
            window_sum += val

        if r >= n - 1:
            max_window_sum = max(max_window_sum, window_sum)

    return max_window_sum

print(maxSubSum([2, 3, 3, 9, 4, 5, 8, 4], 4)) # 26
print(maxSubSum([4, 2, 1, 6, 2], 4)) # 13
print(maxSubSum([], 4)) # null

# Given an array, return true if there are two elements
# within a window of size k that are equal.

def equalElements(arr, k):
    window = set()
    l = 0

    for r, n in enumerate(arr):
        if r - l + 1 > k:
            window.remove(arr[l])
            l += 1
        if n in window:
            return True
        
        window.add(n)

    return False

print(equalElements([1,2,3,2,3,3], 2)) # True
print(equalElements([1,2,3,2,3,3], 3)) # True