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

![image](https://user-images.githubusercontent.com/11992095/195976783-29e5f88d-20dc-4e6f-822c-109cac983f57.png)

[source](https://www.geeksforgeeks.org/data-structures/linked-list/)

<details>
<summary>View contents</summary>

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

### Breath First Traversals

<details>
<summary>View contents</summary>

<img width="848" alt="image" src="https://user-images.githubusercontent.com/11992095/195859575-520cccdc-621e-4de6-ad4c-4f8185a8f30d.png">
    
<img width="1187" alt="image" src="https://user-images.githubusercontent.com/11992095/195860975-8d448e5a-0635-455d-b219-9028dcf58574.png">


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

**Implementation:**

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

</details>

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
