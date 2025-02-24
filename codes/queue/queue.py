# FIFO - First In First Out
# Enqueue - O(1)
# Dequeue - O(1)

from collections import deque

# dq = deque()
dq = deque([1,2,3])

# append to the right(end)
dq.append(4)
print(dq) # deque([1,2,3,4])

# append to the left(beginning)
dq.appendleft(0)
print(dq) # deque([0,1,2,3,4])

# pop from the right(end)
print(dq.pop()) # 4
print(dq) # deque([0,1,2,3])

# pop from the left(beginning)
print(dq.popleft()) # 0
print(dq) # deque([1,2,3])

# extend to the right(end)
dq.extend([4,5])
print(dq) # deque([1,2,3,4,5])

# extend to the left(beginning)
dq.extendleft([-1,0]) 
print(dq) # deque([-1,0,1,2,3,4,5])

# for queue, we will use dq.append() and dq.popleft()