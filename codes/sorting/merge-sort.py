# MergeSort - Divide and Merge
# TC: O(nlogn), SC: worst - O(n), best - O(1)

def mergeSort(nums):
    def merge(arr, l, m, r):
        left, right = arr[l:m+1], arr[m+1:r+1]
        i, j, k = l, 0, 0

        while j < len(left) and k < len(right):
            if left[j] < right[k]:
                nums[i] = left[j]
                j += 1
            else:
                nums[i] = right[k]
                k += 1
            i += 1

        while j < len(left):
            nums[i] = left[j]
            j += 1
            i += 1

        while k < len(right):
            nums[i] = right[k]
            k += 1
            i += 1

    def divide(arr, l, r):
        if l == r:
            return arr

        m = (l + r) // 2
        divide(arr, l, m)
        divide(arr, m + 1, r)

        print('merge', arr, l, m, r)
        merge(arr, l, m, r)

        return arr

    return divide(nums, 0, len(nums) - 1)

nums = [5, 2, 3, 1]
print(mergeSort(nums))