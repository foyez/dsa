# https://www.youtube.com/watch?v=MHf6awe89xw

def binary_search(items, target):
    def recursive_search(left, right):
        if left > right:
            return False

        mid = left + right

        if target == items[mid]:
            return True
        elif target > items[mid]:
            return recursive_search(mid + 1, right)
        else:
            return recursive_search(left, mid - 1)

    left, right = 0, len(items) - 1

    return recursive_search(left, right)

numbers = [1,2,3,4,5,6,7,8,9,10]

print(binary_search(numbers, 3))
print(binary_search(numbers, 11))
print(binary_search(numbers, 0))