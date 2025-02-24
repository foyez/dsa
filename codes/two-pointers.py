# Check if an array is a palindrome.
def isPalindrome(arr):
    l, r = 0, len(arr) - 1

    while l < r:
        if arr[l] != arr[r]:
            return False
        l += 1
        r -= 1

    return True

print(isPalindrome([1,2,7,7,2,1]))
print(isPalindrome([1,2,7,2,1]))

# Given a sorted input array, return the two indices of two elements
# which sum up to the target value.
def targetSum(arr, target):
    l, r = 0, len(arr) - 1

    while l < r:
        sum = arr[l] + arr[r]
        if sum == target:
            return l, r
        elif sum > target:
            r -= 1
        else:
            l += 1

print(targetSum([-1,2,3,4,8,9], 7))