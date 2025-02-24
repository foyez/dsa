import heapq

def min_heap(items):
    heapq.heapify(items)

    heapq.heappush(items, 10)

    print(heapq.nlargest(2, items)) # n largest value
    print(heapq.nsmallest(2, items)) # n smallest value

    print([heapq.heappop(items) for _ in range(len(items))])

def max_heap(items):
    items = [-v for v in items]
    heapq.heapify(items)

    heapq.heappush(items, -10)

    print([heapq.heappop(items) for _ in range(len(items))])

min_heap([1,3,2,5,7])
max_heap([1,3,2,5,7])