# https://www.youtube.com/watch?v=MHf6awe89xw

def binary_search(items, target):
    left, right = 0, len(items) - 1

    while left <= right:
        mid = (left + right) // 2

        if (target == items[mid]):
            return True
        elif (items[mid] < target):
            left = mid + 1
        else:
            right = mid - 1
    
    return False

# items length 32
# 32 -> 16 -> 8 -> 4 -> 2 -> 1
# 6 steps
# log2^32 = log2^2^5 = 5*log2^2 = 5.1 = 5 ~= 6
# Time Complexity: O(logn)
# Space Complexity: O(1)

# Overflow Case
# INT_MAX + INT_MAX = can't store an integer
# mid = left + (right - left)/2
# mid = (2left + right - left) / 2
# mid = (left + right) / 2

numbers = [1,2,3,4,5,6,7,8,9,10]
print(binary_search(numbers, 3))
print(binary_search(numbers, 11))
print(binary_search(numbers, 0))