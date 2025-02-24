# Pick a pivot & place it in its correct place
# TC: worst - O(n^2), best, avg - O(nlogn), SC: O(logn)

def quickSort(nums):
    def swap(arr, i, j):
        temp = arr[i]
        arr[i] = arr[j]
        arr[j] = temp

    def partition(arr, l, r):
        m = (l + r) // 2
        pivot = arr[m]

        while l <= r:
            while arr[l] < pivot:
                l += 1

            while arr[r] > pivot:
                r -= 1

            if l <= r:
                swap(arr, l, r)
                l += 1
                r -= 1

        return l

    def qs(arr, l, r):
        if l >= r:
            return arr

        index = partition(arr, l, r)
        qs(arr, l, index-1)
        qs(arr, index, r)

        return arr

    return qs(nums, 0, len(nums) - 1)

print(quickSort([5, 2, 3, 1]))
print(quickSort([4, 2, 1, 5, 3]))