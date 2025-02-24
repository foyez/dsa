# Depth-First-Search(DFS) Traversal
# Time Complexity: O(n)

# BST Tre
#              4
#             /  \
#            3    6
#           /    / \
#          2    5   7

from collections import deque

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def bfs(root):
    queue = deque()

    if root:
        queue.append(root)

    level = 0
    while len(queue) > 0:
        print()
        print("level: ", level)
        for _ in range(len(queue)):
            curr = queue.popleft()
            print(curr.val, end=" ")

            if curr.left:
                queue.append(curr.left)

            if curr.right:
                queue.append(curr.right)

        level += 1

if __name__ == "__main__":
    root = Node(4)
    root.left = Node(3)
    root.left.left = Node(2)
    root.right = Node(6)
    root.right.left = Node(5)
    root.right.right = Node(7)

    bfs(root)