# push the maximum to the last by adjacent swaps
# TC: worst - O(n^2), best - O(n), SC: O(1)

def bubbleSort(nums):
    for i in range(len(nums) - 1, -1, -1):
        didSwap = False

        for j in range(0, i):
            if nums[j] > nums[j+1]:
                temp = nums[j]
                nums[j] = nums[j+1]
                nums[j+1] = temp

                didSwap = True

        if not didSwap:
            break

    return nums

l = [5, 2, 3, 1]
print(bubbleSort(l))