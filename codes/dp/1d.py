# 1-Dimension
# Fibonacci
# F(0)=0, F(1)=1
# F(n)=F(n-1) + F(n-2)
# F(2) = F(1) + F(0) = 1+0 = 1
# nth Fibonacci
# Time: O(2^n)
def bruteForce(n):
    if n <= 1:
        return n
    return bruteForce(n-1) + bruteForce(n-2)

print(bruteForce(10)) # 55

# memoization
# top down
# Time and space: O(n)
def memoization(n, cache={}):
    if n <= 1:
        return n
    if n in cache:
        return cache[n]
    cache[n] = memoization(n-1) + memoization(n-2)
    return cache[n]

print(memoization(10)) # 55

# bottom up
# Time: O(n)
# Space: O(1)
def dynamic(n):
    if n <= 1:
        return n

    dp = [0, 1]
    i = 2

    while i <= n:
        tmp = dp[1]
        dp[1] = dp[0] + dp[1]
        dp[0] = tmp
        i += 1

    return dp[1]

print(dynamic(10)) # 55