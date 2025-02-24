# Take an element and place it in correct orders
# TC: worst - O(n^2), best - O(n), SC: O(1)

def insertionSort(nums):
    for i in range(len(nums)):
        j = i
        while j > 0 and nums[j-1] > nums[j]:
            temp = nums[j-1]
            nums[j-1] = nums[j]
            nums[j] = temp

            j -= 1

    return nums

l = [5, 2, 3, 1]
print(insertionSort(l))