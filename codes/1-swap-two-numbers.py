

# solution 1: Third variable
def thirdVariable(a, b):
    t = a
    a = b
    b = t
    return a, b


# solution 2: addition & subtraction
def subAdd(a, b):
    a = a + b  # 10 + 20 = 30
    b = a - b  # 30 - 20 = 10
    a = a - b  # 30 - 10 = 20
    return a, b


# solution 3: multiplication & division
def mulDiv(a, b):
    a = a*b  # 10 * 20 = 200
    b = a/b  # 200 / 20 = 10
    a = a/b  # 200 / 10 = 20
    return a, b


# solution 4: bitwise XOR(^)
def bitwiseXOR(a, b):
    a = a ^ b  # 01010 ^ 10100 = 11110 = 30
    b = a ^ b  # 11110 ^ 10100 = 01010 = 10
    a = a ^ b  # 11110 ^ 01010 = 10100 = 20
    return a, b


# solution 5: single line
def _1loc(a, b):
    a, b = b, a
    return a, b


if __name__ == '__main__':
    a, b = 10, 20

    print(thirdVariable(a, b))
    print(subAdd(a, b))
    print(mulDiv(a, b))
    print(bitwiseXOR(a, b))
    print(_1loc(a, b))
    print(a, b)
