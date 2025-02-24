# Depth-First-Search(DFS) Traversal
# Time Complexity: O(n)

# BST Tre
#              4
#             /  \
#            3    6
#           /    / \
#          2    5   7

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def preorder(root):
    if not root:
        return

    print(root.val, end=" ")
    inorder(root.left)
    inorder(root.right)

def inorder(root):
    if not root:
        return

    inorder(root.left)
    print(root.val, end=" ")
    inorder(root.right)

def reverseorder(root):
    if not root:
        return

    reverseorder(root.right)
    print(root.val, end=" ")
    reverseorder(root.left)

def postorder(root):
    if not root:
        return

    inorder(root.left)
    inorder(root.right)
    print(root.val, end=" ")

if __name__ == "__main__":
    root = Node(4)
    root.left = Node(3)
    root.left.left = Node(2)
    root.right = Node(6)
    root.right.left = Node(5)
    root.right.right = Node(7)

    print("Pre Order:", end=" ")
    preorder(root)
    print()

    print("In Order:", end=" ")
    inorder(root)
    print()

    print("Reverse Sorting Order:", end=" ")
    reverseorder(root)
    print()

    print("Post Order:", end=" ")
    postorder(root)
