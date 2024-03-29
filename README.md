# A note on Data Structures & Algorithms

## Big-O Notation

> Used to classify algorithms according to how their **run time** or **space** requirements grow as the input size grows.

<details>
<summary>View contents</summary>

#### Time Complexity

> analyze the runtime as the size of the inputs increases.

- Arithmetic operations are constant.
- Variable assignment is constant.
- Accessing elements in an array (by index) or object (by key) is constant.
- In a loop, the complexity is the length of the loop times.

#### Space Complexity

> how much additional memory do we need to allocate.

- Most primitives (booleans, numbers, undefined, null) are constant space.
- Strings require O(n) space (where n is the string length)
- Reference types are generally O(n), where n is the length (for arrays) or the number of keys (for objects)

#### O (Big Oh), Ω (Big Omega) and Θ (Big Theta)

- Big oh (O) - defines the worst case. e.g.: O(n)
- Big Omega (Ω) - defines the best case. e.g.: Ω(1)
- Big Theta (Θ) - when best case and worst case are same. e.g.: Θ(1)

#### Big-O Complexity Chart

![Big O Complexity Chart](assets/big-o/big-o-complexity-chart.jpg)

source: [https://www.bigocheatsheet.com/](https://www.bigocheatsheet.com/)

#### Big-O list

- ✅ **O(1) Constant Time:** no loops
- ✅ **O(logN) Logarithmic:** usually searching algorithms have log(n) if they are sorted (Binary Search) [size 8 -> 3 operations (log2^8), size 16 -> 4 operations (log2^16)]
- ✅ **O(n) Linear Time:** for, while loops
- ✅ **O(n \* logN):** Log Linear - usually Sorting algorithms
- ✅ **O(n^2) Quadratic Time:** every element in a collection needs to be compared to every other element. Two nested loops
- ✅ **O(2^n) Exponential Time:** recursive algorithms that solve a problem of size N
- ✅ **O(n!) Factorial Time:** Run a loop for every element
- ✅ **Two separate inputs:** O(a + b) or O(a \* b)

#### Common Data Structure Operations

![Common Data Structure Operations](assets/big-o/common-ds-ops.jpg)

source: [https://www.bigocheatsheet.com/](https://www.bigocheatsheet.com/)

#### Array Sorting Algorithms

![Array Sorting Algorithms](assets/big-o/array-sorting-algs.jpg)

source: [https://www.bigocheatsheet.com/](https://www.bigocheatsheet.com/)

</details>

## Linked List

> A linked list consists of nodes where each node contains a data field and a reference(link) to the next node in the list.

<details>
<summary>View contents</summary>

![image](https://user-images.githubusercontent.com/11992095/195976783-29e5f88d-20dc-4e6f-822c-109cac983f57.png)
[source](https://www.geeksforgeeks.org/data-structures/linked-list/)

### Linked List Basic operations

```py

from typing import Optional, Tuple
from typing_extensions import Self


class Node:
    def __init__(self, val: int = 0, next: Optional[Self] = None):
        self.val = val
        self.next = next


class LinkedListCrud:
    def __init__(self) -> None:
        # head: 1 -> 2
        self.head = Node(1)
        self.head.next = Node(2)

    # detect loop
    def detectLoop(self) -> Optional[Node]:
        slow = self.head
        fast = self.head

        while fast and fast.next and slow != fast:
            slow = slow.next
            fast = fast.next.next

        if slow and slow == fast:
            print("Loop detected")
            return slow
        else:
            print("Loop doesn't exists")
            return None

    # initiate loop
    def initiateLoop(self, last: Node, middle: Node) -> None:
        last.next = middle

    # Find middle
    def findMiddle(self) -> Node:
        slow = self.head
        fast = self.head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        return slow

    # Reverse the linked list
    def reverseLL(self):
        prev = None
        curr = self.head

        while curr:
            next = curr.next
            curr.next = prev
            prev = curr
            curr = next

        self.head = prev

    # Delete a node
    def deleteNode(self, index: int):
        dummy = Node(next=self.head)
        prev, curr = dummy, self.head
        count = -1

        while curr and count != index-1:
            prev = curr
            curr = curr.next
            count += 1

        if count+1 != index:
            print("Index {} doesn't exists".format(index))
            return

        prev.next = curr.next
        self.head = dummy.next

    # Insert at a node
    def insertAt(self, index: int, val: int) -> None:
        dummy = Node(next=self.head)
        prev, curr = dummy, self.head
        count = -1

        while curr and count != index-1:
            prev = curr
            curr = curr.next
            count += 1

        if count+1 != index:
            print("Index {} doesn't exists".format(index))
            return

        newNode = Node(val=val)
        newNode.next = curr
        prev.next = newNode
        self.head = dummy.next

        # Insert at the end
    def insertAtEnd(self, val: int) -> None:
        curr = self.head

        while curr and curr.next:
            curr = curr.next
        curr.next = Node(val)

    # Insert at the beginning
    def insertAtBeginning(self, val: int) -> None:
        dummy = Node(val)
        dummy.next = self.head
        self.head = dummy

    # get last node
    def getLastNode(self) -> Node:
        curr = self.head

        while curr and curr.next:
            curr = curr.next

        return curr

    # Print the linked list
    def printLL(self, node=None, msg: str = ""):
        if not self.head:
            print("Linked List is empty.")

        curr = node if node else self.head
        if msg:
            print(msg+":", end=" ")

        while curr:
            print(curr.val, end=" ")
            curr = curr.next
        print()


if __name__ == "__main__":
    ll = LinkedListCrud()

    # crud operations
    ll.printLL(msg="Before Insert")
    ll.insertAtBeginning(5)
    ll.printLL(msg="After inserting 5 at beginning")
    ll.insertAtEnd(100)
    ll.printLL(msg="After inserting 100 at end")
    ll.insertAt(4, 200)
    ll.printLL(msg="After inserting 200 at 4th or last index")
    ll.insertAt(0, 50)
    ll.printLL(msg="After inserting 200 at 0 or 1st index")
    ll.insertAt(2, 46)
    ll.printLL(msg="After inserting 46 at 2nd index")
    ll.deleteNode(0)
    ll.printLL(msg="After deleting beginning node")
    ll.deleteNode(5)
    ll.printLL(msg="After deleting end node")

    # revers a LL
    ll.reverseLL()
    ll.printLL(msg="After reversing the linked list")

    # find middle
    middleNode = ll.findMiddle()
    ll.printLL(node=middleNode, msg="Middle Node")

    # find loop, remove loop, find length of loop
    lastNode = ll.getLastNode()
    ll.initiateLoop(lastNode, middleNode)
    ll.detectLoop()
```

```
Before Insert: 1 2 
After inserting 5 at beginning: 5 1 2 
After inserting 100 at end: 5 1 2 100 
After inserting 200 at 4th or last index: 5 1 2 100 200 
After inserting 200 at 0 or 1st index: 50 5 1 2 100 200 
After inserting 46 at 2nd index: 50 5 46 1 2 100 200 
After deleting beginning node: 5 46 1 2 100 200 
After deleting end node: 5 46 1 2 100 
After reversing the linked list: 100 2 1 46 5 
Middle Node: 1 46 5 
Loop detected
```

</details>

## Tree

> A tree is non-linear and a hierarchical data structure consisting of a collection of nodes such that each node of the tree stores a value and a list of references to other nodes (the “children”). <sup>[ref](https://www.geeksforgeeks.org/introduction-to-tree-data-structure-and-algorithm-tutorials/)</sup>

<details>
<summary>View contents</summary>

### Binary Tree

> A tree is a non-linear data structure. It has no limitation on the number of children. A binary tree has a limitation as any node of the tree has at most two children: a left and a right child.

<details>
<summary>View contents</summary>

![image](https://user-images.githubusercontent.com/11992095/196828888-d53b98ab-ca50-48d6-a97f-d72de9680fd9.png)


#### Some terminology of Complete Binary Tree:
- Root – Node in which no edge is coming from the parent. Example -node A
- Child – Node having some incoming edge is called child. Example – nodes B, H are the child of A and D respectively.
- Sibling – Nodes having the same parent are sibling. Example- J, K are siblings as they have the same parent E.
- Degree of a node – Number of children of a particular parent. Example- Degree of A is 2 and Degree of H is 1. Degree of L is 0.
- Internal/External nodes – Leaf nodes are external nodes and non leaf nodes are internal nodes.
- Level – Count nodes in a path to reach a destination node. Example- Level of node H is 3 as nodes A, D and H themselves form the path.
- Height – Number of edges to reach the destination node, Root is at height 0. Example – Height of node E is 2 as it has two edges from the root.

#### Properties of Complete Binary Tree:
- A complete binary tree is said to be a proper binary tree where all leaves have the same depth.
- In a complete binary tree number of nodes at depth d is 2d. 
- In a  complete binary tree with n nodes height of the tree is log(n+1).
- All the levels except the last level are completely full.

#### Perfect Binary Tree vs Complete Binary Tree:
A binary tree of height ‘h’ having the maximum number of nodes is a perfect binary tree. 
For a given height h, the maximum number of nodes is 2h+1-1.

A complete binary tree of height h is a proper binary tree up to height h-1, and in the last level element are stored in left to right order.

References:
- [Complete Binary Tree](https://www.geeksforgeeks.org/complete-binary-tree/)
- [Binary Tree Data Structure](https://www.geeksforgeeks.org/binary-tree-data-structure/)

</details>

### Breath First Traversals

<details>
<summary>View contents</summary>

<img width="848" alt="image" src="https://user-images.githubusercontent.com/11992095/195859575-520cccdc-621e-4de6-ad4c-4f8185a8f30d.png">
    
<img width="1187" alt="image" src="https://user-images.githubusercontent.com/11992095/195860975-8d448e5a-0635-455d-b219-9028dcf58574.png">

**Implementation (using queue):**

```py
def bfs(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        queue = [root]

        while queue:
            next_level = []

            for node in queue:
                print(node.val)

                if node.left:
                    next_level.append(node.left)

                if node.right:
                    next_level.append(node.right)

            # Move to the next level
            queue = next_level
```


</details>

### Depth First Traversals

<details>
<summary>View contents</summary>

1. Pre-order

![pre-order](assets/graph/pre-order.png)

2. In-order

![pre-order](assets/graph/in-order.png)

3. Post-order

![pre-order](assets/graph/post-order.png)

source: [data structures and algorithms in python](https://classroom.udacity.com/courses/ud513/lessons/7114284829/concepts/77366995150923)

**Implementation (using recursion):**

```py
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


# left - root - right
def printInOrder(root):
    if root:
        printInOrder(root.left)
        print(root.val, end=" ")
        printInOrder(root.right)


# root - left - right
def printPreOrder(root):
    if root:
        print(root.val, end=" ")
        printPreOrder(root.left)
        printPreOrder(root.right)


# left - right - root
def printPostOrder(root):
    if root:
        printPostOrder(root.left)
        printPostOrder(root.right)
        print(root.val, end=" ")


if __name__ == "__main__":
    root = Node("D")
    root.left = Node("B")
    root.right = Node("E")
    root.left.left = Node("A")
    root.left.right = Node("C")
    root.right.right = Node("F")

    print("Pre Order:", end=" ")
    printPreOrder(root)
    print()

    print("In Order:", end=" ")
    printInOrder(root)
    print()

    print("Post Order:", end=" ")
    printPostOrder(root)
```

**Implementation (using stack):**

```py
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def printPreOrder(root):
    stack = [root]

    while stack:
        node = stack.pop()
        print(node.val)

        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)


if __name__ == '__main__':
    D = Node('D')
    B = Node('B')
    E = Node('E')
    A = Node('A')
    C = Node('C')
    F = Node('F')

    D.left = B
    D.right = E
    B.left = A
    B.right = C
    E.right = F

    printPreOrder(D)
```

</details>

</details>

## Trie

> A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. There are various applications of this data structure, such as autocomplete and spellchecker.

<details>
<summary>View contents</summary>

<img width="520" alt="image" src="https://user-images.githubusercontent.com/11992095/197226342-5440930f-2307-417f-8c76-8f3fa7390353.png">

[source](https://en.wikipedia.org/wiki/Trie#/media/File:Trie_example.svg)

### Implement Trie

```py
class TrieNode:
  def __init__(self):
    self.children = {}
    self.endOfWord = False

class Trie:

    def __init__(self):
      self.root = TrieNode()
        

    def insert(self, word: str) -> None:
      curr = self.root
      
      for ch in word:
        if ch not in curr.children:
          curr.children[ch] = TrieNode()
        curr = curr.children[ch]
        
      curr.endOfWord = True
        

    def search(self, word: str) -> bool:
      curr = self.root
      
      for ch in word:
        if ch not in curr.children:
          return False
        curr = curr.children[ch]
      return curr.endOfWord
        

    def startsWith(self, prefix: str) -> bool:
      curr = self.root
      
      for ch in prefix:
        if ch not in curr.children:
          return False
        curr = curr.children[ch]
        
      return True
      
trie = Trie()
trie.insert("apple")
trie.search("apple") # True
trie.search("app") # False
trie.startsWith("app") # True
```

</details>

## Graph

> Graph is a data structure designed to show relationships between objects.

<details>
<summary>View contents</summary>

The purpose of a graph is to show how different things are connected to one another (also known as network). A graph is similar to a tree.

![graph-node-edge](assets/graph/graph.png)

### BFS (Breath First Search)

<details>
<summary>View contents</summary>

[BFS in geekforgeeks](https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/)

</details>

</details>
