# 7. Trees

## 7.1 Binary Trees

### 7.1.1 Tree Fundamentals

**Definition**: A tree is a hierarchical data structure consisting of nodes connected by edges. Each node contains a value and references to child nodes. A **binary tree** is a tree where each node has at most two children, referred to as left child and right child.

**Key Terminology:**

- **Root**: Topmost node (no parent)
- **Leaf**: Node with no children
- **Parent**: Node with children
- **Child**: Node descended from another node
- **Sibling**: Nodes with same parent
- **Ancestor**: Node's parent, grandparent, etc.
- **Descendant**: Node's children, grandchildren, etc.
- **Subtree**: Tree formed by a node and its descendants
- **Height**: Longest path from node to leaf
- **Depth**: Length of path from root to node
- **Level**: Depth + 1 (root is level 1)

**Visual Representation**:
```
        1          ← Root (Height: 3, Depth: 0, Level: 1)
       / \
      2   3        ← Level 2 (Depth: 1)
     / \   \
    4   5   6      ← Level 3 (Depth: 2)
   /
  7                ← Leaf (Height: 0, Depth: 3, Level: 4)

Height of tree: 3 (edges on longest path from root to leaf)
```

**Properties:**
- Maximum nodes at level L: 2^(L-1)
- Maximum nodes in tree of height h: 2^(h+1) - 1
- Minimum height for n nodes: ⌈log₂(n+1)⌉ - 1

**Real-World Analogies**:
- **Family tree**: Ancestors and descendants
- **File system**: Folders containing files and subfolders
- **Organization chart**: CEO → VPs → Managers → Employees
- **Decision tree**: Yes/no questions leading to outcomes

**Node Structure**:
```python
class TreeNode:
    """Binary tree node"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Creating a simple tree:
#     1
#    / \
#   2   3
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
```

---

### 7.1.2 Tree Traversal Methods

**Definition**: Tree traversal is the process of visiting each node in a tree exactly once in a specific order.

#### Depth-First Search (DFS) Traversals

**1. Inorder Traversal (Left → Root → Right)**

**Definition**: Visit left subtree, then root, then right subtree.

**Use Cases:**
- Get sorted order in BST
- Evaluate expression trees
- Generate infix notation

```python
def inorder_traversal(root):
    """
    Inorder: Left → Root → Right
    
    For BST, produces sorted order.
    """
    if not root:
        return []
    
    result = []
    result.extend(inorder_traversal(root.left))   # Left
    result.append(root.val)                        # Root
    result.extend(inorder_traversal(root.right))  # Right
    
    return result

# Time: O(n) - visit each node once
# Space: O(h) - recursion stack, h = height

# Example tree:
#     4
#    / \
#   2   6
#  / \ / \
# 1  3 5  7

# Inorder: [1, 2, 3, 4, 5, 6, 7] (sorted!)
```

**Iterative Inorder**:
```python
def inorder_iterative(root):
    """Iterative inorder using stack"""
    result = []
    stack = []
    current = root
    
    while current or stack:
        # Go to leftmost node
        while current:
            stack.append(current)
            current = current.left
        
        # Process node
        current = stack.pop()
        result.append(current.val)
        
        # Move to right subtree
        current = current.right
    
    return result

# Time: O(n)
# Space: O(h) - stack
```

---

**2. Preorder Traversal (Root → Left → Right)**

**Definition**: Visit root first, then left subtree, then right subtree.

**Use Cases:**
- Copy tree structure
- Prefix expression evaluation
- Serialize tree

```python
def preorder_traversal(root):
    """
    Preorder: Root → Left → Right
    
    Visits parent before children.
    """
    if not root:
        return []
    
    result = []
    result.append(root.val)                        # Root
    result.extend(preorder_traversal(root.left))   # Left
    result.extend(preorder_traversal(root.right))  # Right
    
    return result

# Time: O(n)
# Space: O(h)

# Example tree:
#     1
#    / \
#   2   3
#  / \
# 4   5

# Preorder: [1, 2, 4, 5, 3]
```

**Iterative Preorder**:
```python
def preorder_iterative(root):
    """Iterative preorder using stack"""
    if not root:
        return []
    
    result = []
    stack = [root]
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        
        # Push right first (so left is processed first)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    
    return result

# Time: O(n)
# Space: O(h)
```

---

**3. Postorder Traversal (Left → Right → Root)**

**Definition**: Visit left subtree, then right subtree, then root.

**Use Cases:**
- Delete tree (delete children before parent)
- Postfix expression evaluation
- Calculate directory sizes

```python
def postorder_traversal(root):
    """
    Postorder: Left → Right → Root
    
    Visits children before parent.
    """
    if not root:
        return []
    
    result = []
    result.extend(postorder_traversal(root.left))   # Left
    result.extend(postorder_traversal(root.right))  # Right
    result.append(root.val)                         # Root
    
    return result

# Time: O(n)
# Space: O(h)

# Example tree:
#     1
#    / \
#   2   3
#  / \
# 4   5

# Postorder: [4, 5, 2, 3, 1]
```

**Iterative Postorder**:
```python
def postorder_iterative(root):
    """Iterative postorder using two stacks"""
    if not root:
        return []
    
    stack1 = [root]
    stack2 = []
    
    while stack1:
        node = stack1.pop()
        stack2.append(node)
        
        if node.left:
            stack1.append(node.left)
        if node.right:
            stack1.append(node.right)
    
    result = []
    while stack2:
        result.append(stack2.pop().val)
    
    return result

# Time: O(n)
# Space: O(n)
```

---

#### Breadth-First Search (BFS) Traversal

**4. Level Order Traversal**

**Definition**: Visit nodes level by level, left to right.

**Use Cases:**
- Find shortest path in unweighted tree
- Print tree by levels
- Serialize tree level by level

```python
from collections import deque

def level_order_traversal(root):
    """
    LeetCode 102: Binary Tree Level Order Traversal
    
    Visit nodes level by level using queue.
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result

# Time: O(n)
# Space: O(w) where w = max width of tree

# Example tree:
#     3
#    / \
#   9  20
#     /  \
#    15   7

# Level order: [[3], [9, 20], [15, 7]]
```

**Traversal Summary**:
```
Tree:     1
         / \
        2   3
       / \
      4   5

Inorder:    [4, 2, 5, 1, 3]  (Left → Root → Right)
Preorder:   [1, 2, 4, 5, 3]  (Root → Left → Right)
Postorder:  [4, 5, 2, 3, 1]  (Left → Right → Root)
Level:      [[1], [2, 3], [4, 5]]  (Level by level)
```

---

### 7.1.3 Common Binary Tree Problems

#### Problem: Maximum Depth

```python
def max_depth(root):
    """
    LeetCode 104: Maximum Depth of Binary Tree
    
    Find height of tree (longest path from root to leaf).
    
    Recursive approach: 1 + max of children's depths
    """
    if not root:
        return 0
    
    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)
    
    return 1 + max(left_depth, right_depth)

# Time: O(n)
# Space: O(h) - recursion stack

# Iterative BFS approach:
def max_depth_iterative(root):
    """Level order traversal, count levels"""
    if not root:
        return 0
    
    queue = deque([root])
    depth = 0
    
    while queue:
        depth += 1
        for _ in range(len(queue)):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return depth
```

---

#### Problem: Same Tree

```python
def is_same_tree(p, q):
    """
    LeetCode 100: Same Tree
    
    Check if two trees are structurally identical with same values.
    """
    # Both null
    if not p and not q:
        return True
    
    # One null, other not
    if not p or not q:
        return False
    
    # Values different
    if p.val != q.val:
        return False
    
    # Check both subtrees
    return (is_same_tree(p.left, q.left) and 
            is_same_tree(p.right, q.right))

# Time: O(min(n, m)) - n and m are tree sizes
# Space: O(min(h1, h2)) - recursion depth
```

---

#### Problem: Invert Binary Tree

```python
def invert_tree(root):
    """
    LeetCode 226: Invert Binary Tree
    
    Mirror the tree (swap left and right subtrees).
    
    Example:
         4              4
       /   \          /   \
      2     7   →    7     2
     / \   / \      / \   / \
    1   3 6   9    9   6 3   1
    """
    if not root:
        return None
    
    # Swap children
    root.left, root.right = root.right, root.left
    
    # Recursively invert subtrees
    invert_tree(root.left)
    invert_tree(root.right)
    
    return root

# Time: O(n)
# Space: O(h)
```

---

#### Problem: Symmetric Tree

```python
def is_symmetric(root):
    """
    LeetCode 101: Symmetric Tree
    
    Check if tree is mirror of itself.
    
    Example:
        1
       / \
      2   2
     / \ / \
    3  4 4  3  → True
    """
    def is_mirror(left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        
        return (left.val == right.val and
                is_mirror(left.left, right.right) and
                is_mirror(left.right, right.left))
    
    return is_mirror(root, root) if root else True

# Time: O(n)
# Space: O(h)
```

---

#### Problem: Diameter of Binary Tree

```python
def diameter_of_binary_tree(root):
    """
    LeetCode 543: Diameter of Binary Tree
    
    Find longest path between any two nodes.
    Path may or may not pass through root.
    
    Example:
          1
         / \
        2   3
       / \
      4   5
    
    Diameter = 3 (path: 4→2→1→3 or 5→2→1→3)
    """
    def height(node):
        if not node:
            return 0
        
        left_height = height(node.left)
        right_height = height(node.right)
        
        # Update diameter at this node
        diameter[0] = max(diameter[0], left_height + right_height)
        
        return 1 + max(left_height, right_height)
    
    diameter = [0]  # Use list to modify in nested function
    height(root)
    return diameter[0]

# Time: O(n)
# Space: O(h)
```

---

#### Problem: Path Sum

```python
def has_path_sum(root, target_sum):
    """
    LeetCode 112: Path Sum
    
    Check if root-to-leaf path exists with sum equal to target.
    """
    if not root:
        return False
    
    # Leaf node
    if not root.left and not root.right:
        return root.val == target_sum
    
    # Check children with remaining sum
    remaining = target_sum - root.val
    return (has_path_sum(root.left, remaining) or 
            has_path_sum(root.right, remaining))

# Time: O(n)
# Space: O(h)

def path_sum_all(root, target_sum):
    """
    LeetCode 113: Path Sum II
    
    Find all root-to-leaf paths with sum equal to target.
    """
    def dfs(node, remaining, path):
        if not node:
            return
        
        path.append(node.val)
        
        # Leaf with target sum
        if not node.left and not node.right and remaining == node.val:
            result.append(path[:])  # Copy path
        else:
            # Continue searching
            dfs(node.left, remaining - node.val, path)
            dfs(node.right, remaining - node.val, path)
        
        path.pop()  # Backtrack
    
    result = []
    dfs(root, target_sum, [])
    return result

# Time: O(n²) - worst case all paths
# Space: O(h)
```

---

#### Problem: Lowest Common Ancestor

```python
def lowest_common_ancestor(root, p, q):
    """
    LeetCode 236: Lowest Common Ancestor of a Binary Tree
    
    Find lowest (deepest) node that has both p and q as descendants.
    
    Example:
          3
         / \
        5   1
       / \ / \
      6  2 0  8
        / \
       7   4
    
    LCA(5, 1) = 3
    LCA(5, 4) = 5 (node can be ancestor of itself)
    """
    # Base case
    if not root or root == p or root == q:
        return root
    
    # Search in subtrees
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    
    # Both found in different subtrees → current node is LCA
    if left and right:
        return root
    
    # Return whichever is not null
    return left if left else right

# Time: O(n)
# Space: O(h)
```

---

#### Problem: Serialize and Deserialize

```python
class Codec:
    """
    LeetCode 297: Serialize and Deserialize Binary Tree
    
    Convert tree to string and back.
    """
    def serialize(self, root):
        """Encode tree to string using preorder"""
        def helper(node):
            if not node:
                vals.append('null')
            else:
                vals.append(str(node.val))
                helper(node.left)
                helper(node.right)
        
        vals = []
        helper(root)
        return ','.join(vals)
    
    def deserialize(self, data):
        """Decode string to tree"""
        def helper():
            val = next(vals)
            if val == 'null':
                return None
            
            node = TreeNode(int(val))
            node.left = helper()
            node.right = helper()
            return node
        
        vals = iter(data.split(','))
        return helper()

# Time: O(n) for both
# Space: O(n)

# Example:
#     1
#    / \
#   2   3
#      / \
#     4   5
# Serialized: "1,2,null,null,3,4,null,null,5,null,null"
```

---

### 7.1.4 Binary Tree Patterns

#### Pattern 1: Recursive DFS Template

```python
def dfs_template(root):
    """
    Generic DFS template for binary trees.
    
    Use for: tree properties, path finding, validation
    """
    # Base case
    if not root:
        return base_value
    
    # Process current node
    current_result = process(root)
    
    # Recurse on children
    left_result = dfs_template(root.left)
    right_result = dfs_template(root.right)
    
    # Combine results
    return combine(current_result, left_result, right_result)
```

#### Pattern 2: Iterative BFS Template

```python
def bfs_template(root):
    """
    Generic BFS template using queue.
    
    Use for: level order, shortest path, breadth-first tasks
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result
```

#### Pattern 3: Path Tracking

```python
def find_paths_template(root, target):
    """
    Template for finding paths in tree.
    
    Use for: path sum, all paths, specific paths
    """
    def dfs(node, path, current_sum):
        if not node:
            return
        
        # Add to path
        path.append(node.val)
        current_sum += node.val
        
        # Check if leaf with target
        if not node.left and not node.right:
            if current_sum == target:
                result.append(path[:])
        else:
            # Continue searching
            dfs(node.left, path, current_sum)
            dfs(node.right, path, current_sum)
        
        # Backtrack
        path.pop()
    
    result = []
    dfs(root, [], 0)
    return result
```

---

## 7.2 Binary Search Trees (BST)

### 7.2.1 BST Fundamentals

**Definition**: A Binary Search Tree is a binary tree with the ordering property: for every node, all values in its left subtree are less than the node's value, and all values in its right subtree are greater than the node's value.

**BST Property**:
```
For every node N:
- All nodes in left subtree < N.val
- All nodes in right subtree > N.val

Example BST:
        5
       / \
      3   7
     / \ / \
    2  4 6  8

Inorder: [2, 3, 4, 5, 6, 7, 8] ← Sorted!
```

**Why BST:**
- Search: O(h) where h = height
- Insert: O(h)
- Delete: O(h)
- Inorder traversal gives sorted order
- Balanced BST: h = log n → O(log n) operations

**Real-World Uses**:
- Database indexing
- Dictionary/map implementations
- Auto-complete systems
- Range queries

---

### 7.2.2 BST Operations

#### Search in BST

```python
def search_bst(root, val):
    """
    LeetCode 700: Search in a Binary Search Tree
    
    Search for value in BST.
    """
    # Not found
    if not root:
        return None
    
    # Found
    if root.val == val:
        return root
    
    # Search left or right
    if val < root.val:
        return search_bst(root.left, val)
    else:
        return search_bst(root.right, val)

# Time: O(h) - h = height
# Space: O(h) - recursion

# Iterative version:
def search_bst_iterative(root, val):
    """Iterative search"""
    while root:
        if root.val == val:
            return root
        elif val < root.val:
            root = root.left
        else:
            root = root.right
    return None

# Time: O(h)
# Space: O(1)
```

---

#### Insert into BST

```python
def insert_into_bst(root, val):
    """
    LeetCode 701: Insert into a Binary Search Tree
    
    Insert value maintaining BST property.
    """
    # Empty tree or found insertion point
    if not root:
        return TreeNode(val)
    
    # Go left or right
    if val < root.val:
        root.left = insert_into_bst(root.left, val)
    else:
        root.right = insert_into_bst(root.right, val)
    
    return root

# Time: O(h)
# Space: O(h)

# Iterative version:
def insert_into_bst_iterative(root, val):
    """Iterative insert"""
    if not root:
        return TreeNode(val)
    
    current = root
    while True:
        if val < current.val:
            if not current.left:
                current.left = TreeNode(val)
                break
            current = current.left
        else:
            if not current.right:
                current.right = TreeNode(val)
                break
            current = current.right
    
    return root

# Time: O(h)
# Space: O(1)
```

---

#### Delete from BST

```python
def delete_node(root, key):
    """
    LeetCode 450: Delete Node in a BST
    
    Delete node with value key from BST.
    
    Cases:
    1. Node is leaf → simply remove
    2. Node has one child → replace with child
    3. Node has two children → replace with inorder successor
    """
    if not root:
        return None
    
    # Find node to delete
    if key < root.val:
        root.left = delete_node(root.left, key)
    elif key > root.val:
        root.right = delete_node(root.right, key)
    else:
        # Found node to delete
        
        # Case 1 & 2: 0 or 1 child
        if not root.left:
            return root.right
        if not root.right:
            return root.left
        
        # Case 3: 2 children
        # Find inorder successor (smallest in right subtree)
        successor = root.right
        while successor.left:
            successor = successor.left
        
        # Replace with successor's value
        root.val = successor.val
        
        # Delete successor
        root.right = delete_node(root.right, successor.val)
    
    return root

# Time: O(h)
# Space: O(h)
```

---

### 7.2.3 BST Validation and Properties

#### Validate BST

```python
def is_valid_bst(root):
    """
    LeetCode 98: Validate Binary Search Tree
    
    Check if tree satisfies BST property.
    
    Key insight: Not enough to check node.left < node < node.right
    Must ensure ALL left subtree < node < ALL right subtree
    """
    def validate(node, min_val, max_val):
        if not node:
            return True
        
        # Check current node's value is in valid range
        if not (min_val < node.val < max_val):
            return False
        
        # Check subtrees with updated ranges
        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))
    
    return validate(root, float('-inf'), float('inf'))

# Time: O(n)
# Space: O(h)

# Alternative: Inorder traversal should be sorted
def is_valid_bst_inorder(root):
    """Check if inorder is sorted"""
    def inorder(node):
        if not node:
            return True
        
        if not inorder(node.left):
            return False
        
        if prev[0] is not None and prev[0] >= node.val:
            return False
        prev[0] = node.val
        
        return inorder(node.right)
    
    prev = [None]
    return inorder(root)
```

---

#### Kth Smallest Element

```python
def kth_smallest(root, k):
    """
    LeetCode 230: Kth Smallest Element in a BST
    
    Find kth smallest element (1-indexed).
    
    Approach: Inorder traversal gives sorted order.
    """
    def inorder(node):
        if not node:
            return
        
        inorder(node.left)
        
        count[0] += 1
        if count[0] == k:
            result[0] = node.val
            return
        
        inorder(node.right)
    
    count = [0]
    result = [None]
    inorder(root)
    return result[0]

# Time: O(h + k) - worst case O(n)
# Space: O(h)

# Iterative version:
def kth_smallest_iterative(root, k):
    """Iterative inorder"""
    stack = []
    current = root
    count = 0
    
    while current or stack:
        while current:
            stack.append(current)
            current = current.left
        
        current = stack.pop()
        count += 1
        
        if count == k:
            return current.val
        
        current = current.right
    
    return None
```

---

#### Range Sum BST

```python
def range_sum_bst(root, low, high):
    """
    LeetCode 938: Range Sum of BST
    
    Sum all values in range [low, high].
    
    Optimization: Use BST property to prune branches.
    """
    if not root:
        return 0
    
    total = 0
    
    # Add current if in range
    if low <= root.val <= high:
        total += root.val
    
    # Prune left subtree if root < low
    if root.val > low:
        total += range_sum_bst(root.left, low, high)
    
    # Prune right subtree if root > high
    if root.val < high:
        total += range_sum_bst(root.right, low, high)
    
    return total

# Time: O(n) worst case, but prunes branches
# Space: O(h)
```

---

### 7.2.4 Balanced BSTs

**Definition**: A balanced BST maintains height ≈ log n to guarantee O(log n) operations.

**Common Self-Balancing BSTs:**
- **AVL Tree**: Strictly balanced (height diff ≤ 1)
- **Red-Black Tree**: Loosely balanced (used in most libraries)
- **B-Tree**: Generalizes BST for disk storage

#### Check if Balanced

```python
def is_balanced(root):
    """
    LeetCode 110: Balanced Binary Tree
    
    Check if height-balanced (left and right subtree heights
    differ by at most 1 for every node).
    """
    def check_height(node):
        if not node:
            return 0
        
        left_height = check_height(node.left)
        if left_height == -1:
            return -1  # Left subtree not balanced
        
        right_height = check_height(node.right)
        if right_height == -1:
            return -1  # Right subtree not balanced
        
        # Check if current node balanced
        if abs(left_height - right_height) > 1:
            return -1
        
        return 1 + max(left_height, right_height)
    
    return check_height(root) != -1

# Time: O(n)
# Space: O(h)
```

---

## 7.3 Tries (Prefix Trees)

### 7.3.1 Trie Fundamentals

**Definition**: A trie (pronounced "try") is a tree-like data structure used to store strings where each node represents a character. Tries are used for efficient storage and retrieval of strings, particularly for prefix-based operations.

**Structure**:
```
Storing: ["cat", "car", "card", "dog"]

        root
       /    \
      c      d
      |      |
      a      o
     / \     |
    r   t    g*
    |
    d*

* = end of word
```

**Key Properties:**
- Each path from root to node represents a prefix
- Nodes marked as "end of word" represent complete strings
- Common prefixes share paths
- Height ≤ longest word length

**Why Trie:**
- Insert word: O(L) where L = word length
- Search word: O(L)
- Search prefix: O(L)
- Space efficient for large sets with common prefixes
- Supports autocomplete, spell check, IP routing

**Real-World Uses**:
- Autocomplete/typeahead
- Spell checkers
- IP routing tables
- Dictionary implementations
- DNA sequence storage

---

### 7.3.2 Trie Implementation

```python
class TrieNode:
    """Node in trie"""
    def __init__(self):
        self.children = {}  # char → TrieNode
        self.is_end_of_word = False

class Trie:
    """
    LeetCode 208: Implement Trie (Prefix Tree)
    
    Trie supporting insert, search, and prefix search.
    """
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        """
        Insert word into trie.
        
        Time: O(L) where L = len(word)
        Space: O(L) in worst case
        """
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end_of_word = True
    
    def search(self, word):
        """
        Search for exact word in trie.
        
        Time: O(L)
        Space: O(1)
        """
        node = self.root
        
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return node.is_end_of_word
    
    def starts_with(self, prefix):
        """
        Check if any word has this prefix.
        
        Time: O(L)
        Space: O(1)
        """
        node = self.root
        
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return True

# Example usage:
trie = Trie()
trie.insert("apple")
print(trie.search("apple"))    # True
print(trie.search("app"))      # False
print(trie.starts_with("app")) # True
trie.insert("app")
print(trie.search("app"))      # True
```

---

### 7.3.3 Trie Problems

#### Design Add and Search Words

```python
class WordDictionary:
    """
    LeetCode 211: Design Add and Search Words Data Structure
    
    Supports adding words and searching with wildcards.
    '.' matches any character.
    """
    def __init__(self):
        self.root = TrieNode()
    
    def add_word(self, word):
        """Add word to dictionary"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
    
    def search(self, word):
        """
        Search with wildcards.
        '.' matches any single character.
        """
        def dfs(node, i):
            if i == len(word):
                return node.is_end_of_word
            
            char = word[i]
            
            if char == '.':
                # Try all children
                for child in node.children.values():
                    if dfs(child, i + 1):
                        return True
                return False
            else:
                if char not in node.children:
                    return False
                return dfs(node.children[char], i + 1)
        
        return dfs(self.root, 0)

# Time: 
# - add: O(L)
# - search: O(L) best case, O(26^L) worst with all wildcards
# Space: O(total characters stored)

# Example:
wd = WordDictionary()
wd.add_word("bad")
wd.add_word("dad")
wd.add_word("mad")
print(wd.search("pad"))   # False
print(wd.search("bad"))   # True
print(wd.search(".ad"))   # True
print(wd.search("b.."))   # True
```

---

#### Word Search II

```python
class Solution:
    """
    LeetCode 212: Word Search II
    
    Find all words from list that exist in 2D board.
    
    Optimized with trie: search multiple words simultaneously.
    """
    def find_words(self, board, words):
        # Build trie from word list
        root = TrieNode()
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end_of_word = True
            node.word = word  # Store word at end node
        
        def dfs(row, col, node, path):
            char = board[row][col]
            
            # Check if char in trie
            if char not in node.children:
                return
            
            next_node = node.children[char]
            
            # Found word
            if next_node.is_end_of_word:
                result.add(next_node.word)
            
            # Mark visited
            board[row][col] = '#'
            
            # Explore neighbors
            for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                nr, nc = row + dr, col + dc
                if (0 <= nr < len(board) and 
                    0 <= nc < len(board[0]) and 
                    board[nr][nc] != '#'):
                    dfs(nr, nc, next_node, path + char)
            
            # Restore
            board[row][col] = char
        
        result = set()
        
        # Try starting from each cell
        for i in range(len(board)):
            for j in range(len(board[0])):
                dfs(i, j, root, "")
        
        return list(result)

# Time: O(m × n × 4^L) where L = max word length
# Space: O(total characters in words)
```

---

#### Longest Common Prefix

```python
def longest_common_prefix_trie(strs):
    """
    Using trie to find longest common prefix.
    
    Build trie, traverse from root until:
    - Node has multiple children (branching)
    - Node is end of word
    """
    if not strs:
        return ""
    
    # Build trie
    root = TrieNode()
    for word in strs:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
    
    # Find LCP
    prefix = []
    node = root
    
    while len(node.children) == 1 and not node.is_end_of_word:
        char = list(node.children.keys())[0]
        prefix.append(char)
        node = node.children[char]
    
    return ''.join(prefix)

# Time: O(S) where S = sum of all string lengths
# Space: O(S)

# Example:
strs = ["flower", "flow", "flight"]
print(longest_common_prefix_trie(strs))  # "fl"
```

---

### 7.3.4 Trie vs Hash Table

| Feature | Trie | Hash Table |
|---------|------|------------|
| **Search word** | O(L) | O(L) average |
| **Prefix search** | O(L) | O(N×L) - check all |
| **Space** | More (tree structure) | Less |
| **Autocomplete** | Efficient | Inefficient |
| **Sorted output** | Yes (traverse) | No |
| **Best for** | Prefix operations | Exact lookups |

---

## Practice Questions

### Fill in the Gaps

1. Inorder traversal of a BST produces values in ________ order.
2. The maximum number of nodes at level L is ________.
3. A balanced BST has height approximately ________.
4. The time complexity of searching in a trie is ________ where L is word length.
5. To delete a node with two children in BST, replace with ________ successor.

### True or False

1. Preorder traversal visits root before children. **[T/F]**
2. Every binary tree is a BST. **[T/F]**
3. Tries are more space-efficient than hash tables for single word lookup. **[T/F]**
4. Level order traversal uses a queue. **[T/F]**
5. Postorder traversal is used to delete trees. **[T/F]**

### Multiple Choice

1. Which traversal gives sorted order in BST?
   - A) Preorder
   - B) Inorder
   - C) Postorder
   - D) Level order

2. Time to insert into balanced BST with n nodes?
   - A) O(1)
   - B) O(log n)
   - C) O(n)
   - D) O(n log n)

3. Tries are best for:
   - A) Exact word lookup
   - B) Prefix matching
   - C) Numeric data
   - D) Sorting

### Code Challenge

```python
def right_side_view(root):
    """
    LeetCode 199: Binary Tree Right Side View
    
    Return values visible from right side (rightmost node at each level).
    
    Example:
       1
      / \
     2   3
      \   \
       5   4
    
    Output: [1, 3, 4]
    
    Implement using level order traversal.
    """
    # Your code here
    pass
```

---

## Answers

<details>
<summary><strong>View Answers</strong></summary>

### Fill in the Gaps

1. **sorted** (ascending)
2. **2^(L-1)**
3. **log n**
4. **O(L)**
5. **inorder** (smallest in right subtree)

### True or False

1. **True** - Root → Left → Right
2. **False** - BST has ordering property
3. **False** - Tries use more space due to tree structure
4. **True** - BFS uses queue
5. **True** - Delete children before parent

### Multiple Choice

1. **B** - Inorder: Left → Root → Right
2. **B** - O(log n) in balanced BST
3. **B** - Tries excel at prefix operations

### Code Challenge Answer

```python
def right_side_view(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        
        for i in range(level_size):
            node = queue.popleft()
            
            # Rightmost node of level
            if i == level_size - 1:
                result.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return result

# Time: O(n)
# Space: O(w) where w = max width
```

</details>

---

## LeetCode Problems (NeetCode.io)

### Binary Trees - Easy ✅
- 94. Binary Tree Inorder Traversal
- 100. Same Tree
- 101. Symmetric Tree
- 104. Maximum Depth of Binary Tree
- 111. Minimum Depth of Binary Tree
- 226. Invert Binary Tree
- 543. Diameter of Binary Tree
- 572. Subtree of Another Tree

### Binary Trees - Medium 🟨
- 102. Binary Tree Level Order Traversal (IMPORTANT)
- 103. Binary Tree Zigzag Level Order
- 105. Construct Binary Tree from Preorder and Inorder
- 113. Path Sum II
- 114. Flatten Binary Tree to Linked List
- 199. Binary Tree Right Side View
- 230. Kth Smallest Element in BST
- 236. Lowest Common Ancestor (IMPORTANT)
- 297. Serialize and Deserialize Binary Tree (IMPORTANT)
- 662. Maximum Width of Binary Tree

### Binary Trees - Hard 🔴
- 124. Binary Tree Maximum Path Sum
- 297. Serialize and Deserialize Binary Tree

### BST - Easy ✅
- 98. Validate Binary Search Tree
- 700. Search in a Binary Search Tree
- 938. Range Sum of BST

### BST - Medium 🟨
- 98. Validate Binary Search Tree (IMPORTANT)
- 230. Kth Smallest Element in BST
- 450. Delete Node in a BST
- 701. Insert into a Binary Search Tree

### Tries - Medium 🟨
- 208. Implement Trie (IMPORTANT)
- 211. Design Add and Search Words
- 648. Replace Words
- 676. Implement Magic Dictionary

### Tries - Hard 🔴
- 212. Word Search II (VERY IMPORTANT)

---

*Continue to: [8. Heaps & Priority Queues →](08-heaps.md)*