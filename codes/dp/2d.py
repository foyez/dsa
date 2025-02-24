# 2-Dimension
# Count paths
# Q: Count the number of unique paths from
# the top left to the bottom right. You are only
# allowed to move down or to the right.
# Brute force: Time and Space: O(2^(n+m))
def bruteForce(r, c, rows, cols):
    if r == rows or c == cols:
        return 0
    if r == rows-1 and c == cols-1:
        return 1

    return bruteForce(r+1, c, rows, cols) + bruteForce(r, c+1, rows, cols)
print(bruteForce(0, 0, 4, 4))

# memoization
# top down
# Time and space: O(n+m)
def memoization(r, c, rows, cols, cache):
    if r == rows or c == cols:
        return 0
    if r == rows-1 and c == cols-1:
        return 1
    if cache[r][c] > 0:
        return cache[r][c]

    cache[r][c] = memoization(r+1, c, rows, cols, cache) + memoization(r, c+1, rows, cols, cache)
    return cache[r][c]
print(memoization(0, 0, 4, 4, [[0] * 4 for i in range(4)]))

# Bottom up
# Time: O(n*m), Space: O(m), where m is num of cols
def dynamic(rows, cols):
    prev_row = [0] * cols

    for _r in range(rows-1, -1, -1):
        cur_row = [0] * cols
        cur_row[cols-1] = 1
        for c in range(cols-2, -1, -1):
            cur_row[c] = cur_row[c+1] + prev_row[c]
        prev_row = cur_row
    return prev_row[0]

print(dynamic(4, 4))