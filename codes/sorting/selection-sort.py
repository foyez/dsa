# Selection Sort
# select minimum and swap
# TC: O(n^2), SC: O(1)

def selectionSort(nums):
    for i in range(len(nums) - 1):
        min_index = i

        for j in range(i, len(nums)):
            if nums[min_index] > nums[j]:
                min_index = j

        temp = nums[min_index]
        nums[min_index] = nums[i]
        nums[i] = temp

    return nums

l = [5,2,3,1]
print(selectionSort(l))