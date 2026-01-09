# 10. Graphs

## 10.1 Graph Fundamentals

### 10.1.1 Graph Definition

**Definition**: A graph G = (V, E) is a data structure consisting of a set of vertices (nodes) V and a set of edges E that connect pairs of vertices. Graphs represent relationships and connections between entities.

**Components:**
- **Vertex (Node)**: An entity in the graph
- **Edge**: Connection between two vertices
- **Path**: Sequence of vertices connected by edges
- **Cycle**: Path that starts and ends at same vertex
- **Degree**: Number of edges connected to a vertex

**Types of Graphs:**

1. **Directed vs Undirected**
```
Undirected:          Directed (Digraph):
   A --- B              A --> B
   |     |              |     ^
   |     |              v     |
   C --- D              C --> D

Edge (A,B) = (B,A)   Edge (A,B) ≠ (B,A)
```

2. **Weighted vs Unweighted**
```
Unweighted:          Weighted:
   A --- B              A --5-- B
   |     |              |       |
   |     |              3       2
   C --- D              C --4-- D

All edges equal      Edges have costs
```

3. **Cyclic vs Acyclic**
```
Cyclic:              Acyclic (DAG):
   A --- B              A --> B
   |     |              |     |
   |     |              v     v
   C --- D              C     D
   ^     |
   |_____|

Contains cycles      No cycles (DAG = Directed Acyclic Graph)
```

4. **Connected vs Disconnected**
```
Connected:           Disconnected:
   A --- B              A --- B    E
   |     |              |     |    |
   C --- D              C --- D    F

All vertices         Separate
reachable            components
```

**Graph Properties:**
- **Dense**: |E| ≈ |V|² (many edges)
- **Sparse**: |E| ≈ |V| (few edges)
- **Complete**: Every pair of vertices connected
- **Tree**: Connected acyclic graph with |V| - 1 edges

**Real-World Examples**:
- **Social networks**: People (vertices), friendships (edges)
- **Maps**: Cities (vertices), roads (edges)
- **Web**: Pages (vertices), hyperlinks (edges)
- **Dependencies**: Tasks (vertices), dependencies (edges)

---

### 10.1.2 Graph Representations

#### Adjacency Matrix

**Definition**: 2D array where matrix[i][j] = 1 if edge from i to j exists.

```python
class GraphMatrix:
    """Graph using adjacency matrix"""
    def __init__(self, num_vertices):
        self.V = num_vertices
        self.matrix = [[0] * num_vertices for _ in range(num_vertices)]
    
    def add_edge(self, u, v, directed=False):
        """Add edge from u to v"""
        self.matrix[u][v] = 1
        if not directed:
            self.matrix[v][u] = 1
    
    def has_edge(self, u, v):
        """Check if edge exists"""
        return self.matrix[u][v] == 1
    
    def get_neighbors(self, u):
        """Get all neighbors of vertex u"""
        return [v for v in range(self.V) if self.matrix[u][v] == 1]

# Time: O(1) to add edge, O(1) to check edge, O(V) to get neighbors
# Space: O(V²)

# Example:
#   0 --- 1
#   |     |
#   2 --- 3

g = GraphMatrix(4)
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 3)
g.add_edge(2, 3)

# Matrix:
#   0 1 2 3
# 0[0 1 1 0]
# 1[1 0 0 1]
# 2[1 0 0 1]
# 3[0 1 1 0]
```

**Weighted Graph**:
```python
# Store weight instead of 1
matrix[u][v] = weight
# 0 or infinity for no edge
```

**Pros:**
- O(1) edge lookup
- Good for dense graphs
- Simple implementation

**Cons:**
- O(V²) space even for sparse graphs
- O(V) to iterate neighbors

---

#### Adjacency List

**Definition**: Array of lists where list[i] contains all neighbors of vertex i.

```python
from collections import defaultdict

class GraphList:
    """Graph using adjacency list"""
    def __init__(self):
        self.graph = defaultdict(list)
    
    def add_edge(self, u, v, directed=False):
        """Add edge from u to v"""
        self.graph[u].append(v)
        if not directed:
            self.graph[v].append(u)
    
    def get_neighbors(self, u):
        """Get all neighbors of vertex u"""
        return self.graph[u]
    
    def has_edge(self, u, v):
        """Check if edge exists"""
        return v in self.graph[u]

# Time: O(1) to add edge, O(degree) to check edge/get neighbors
# Space: O(V + E)

# Example:
#   0 --- 1
#   |     |
#   2 --- 3

g = GraphList()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 3)
g.add_edge(2, 3)

# Adjacency list:
# 0: [1, 2]
# 1: [0, 3]
# 2: [0, 3]
# 3: [1, 2]
```

**Weighted Graph**:
```python
# Store tuples (neighbor, weight)
class WeightedGraph:
    def __init__(self):
        self.graph = defaultdict(list)
    
    def add_edge(self, u, v, weight):
        self.graph[u].append((v, weight))
    
    def get_neighbors(self, u):
        return self.graph[u]

# Example
g = WeightedGraph()
g.add_edge(0, 1, 5)
g.add_edge(0, 2, 3)
# 0: [(1, 5), (2, 3)]
```

**Pros:**
- Space efficient for sparse graphs
- Fast neighbor iteration
- Most common representation

**Cons:**
- O(degree) edge lookup
- More complex than matrix

---

#### Edge List

**Definition**: List of all edges as pairs (u, v) or (u, v, weight).

```python
class GraphEdgeList:
    """Graph using edge list"""
    def __init__(self):
        self.edges = []
    
    def add_edge(self, u, v, weight=1):
        self.edges.append((u, v, weight))
    
    def get_edges(self):
        return self.edges

# Example
g = GraphEdgeList()
g.add_edge(0, 1, 5)
g.add_edge(0, 2, 3)
g.add_edge(1, 3, 2)
# edges = [(0, 1, 5), (0, 2, 3), (1, 3, 2)]
```

**Use Cases:**
- Kruskal's MST algorithm
- When you need to sort edges
- When graph structure doesn't matter

---

## 10.2 Graph Traversal

### 10.2.1 Depth-First Search (DFS)

**Definition**: Explore as far as possible along each branch before backtracking. Uses a stack (or recursion).

**Algorithm:**
1. Start at source vertex
2. Mark as visited
3. Recursively visit unvisited neighbors
4. Backtrack when no unvisited neighbors

**Visual Example:**
```
Graph:     1 --- 2
           |     |
           3 --- 4

DFS from 1: 1 → 3 → 4 → 2
(Goes deep before wide)
```

**Recursive Implementation:**
```python
def dfs_recursive(graph, start, visited=None):
    """
    DFS using recursion.
    
    graph: adjacency list (dict of lists)
    start: starting vertex
    visited: set of visited vertices
    """
    if visited is None:
        visited = set()
    
    # Mark as visited
    visited.add(start)
    print(start, end=' ')
    
    # Visit all unvisited neighbors
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)
    
    return visited

# Time: O(V + E) - visit each vertex and edge once
# Space: O(V) - recursion stack + visited set

# Example
graph = {
    1: [2, 3],
    2: [1, 4],
    3: [1, 4],
    4: [2, 3]
}
dfs_recursive(graph, 1)  # Output: 1 2 4 3
```

**Iterative Implementation:**
```python
def dfs_iterative(graph, start):
    """
    DFS using explicit stack.
    """
    visited = set()
    stack = [start]
    
    while stack:
        vertex = stack.pop()
        
        if vertex not in visited:
            visited.add(vertex)
            print(vertex, end=' ')
            
            # Add neighbors to stack
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return visited

# Time: O(V + E)
# Space: O(V)

# Example
dfs_iterative(graph, 1)  # Output: 1 3 4 2 (different order than recursive)
```

**DFS Applications:**
- Detect cycles
- Topological sorting
- Find connected components
- Solve mazes
- Check if path exists

---

### 10.2.2 Breadth-First Search (BFS)

**Definition**: Explore all neighbors at current depth before moving to next depth. Uses a queue.

**Algorithm:**
1. Start at source vertex
2. Add to queue and mark visited
3. Dequeue vertex, process it
4. Enqueue all unvisited neighbors
5. Repeat until queue empty

**Visual Example:**
```
Graph:     1 --- 2
           |     |
           3 --- 4

BFS from 1: 1 → 2 → 3 → 4
(Goes wide before deep)

Level 0: [1]
Level 1: [2, 3]
Level 2: [4]
```

**Implementation:**
```python
from collections import deque

def bfs(graph, start):
    """
    BFS traversal.
    
    Returns visited vertices in BFS order.
    """
    visited = set()
    queue = deque([start])
    visited.add(start)
    result = []
    
    while queue:
        vertex = queue.popleft()
        result.append(vertex)
        
        # Add unvisited neighbors
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result

# Time: O(V + E)
# Space: O(V) - queue + visited set

# Example
graph = {
    1: [2, 3],
    2: [1, 4],
    3: [1, 4],
    4: [2, 3]
}
print(bfs(graph, 1))  # [1, 2, 3, 4]
```

**BFS with Levels:**
```python
def bfs_levels(graph, start):
    """
    BFS tracking distance/level from start.
    """
    visited = {start}
    queue = deque([(start, 0)])  # (vertex, level)
    levels = {start: 0}
    
    while queue:
        vertex, level = queue.popleft()
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, level + 1))
                levels[neighbor] = level + 1
    
    return levels

# Example
levels = bfs_levels(graph, 1)
# {1: 0, 2: 1, 3: 1, 4: 2}
```

**BFS Applications:**
- Shortest path in unweighted graph
- Level order traversal
- Find connected components
- Web crawling
- Social network distance

---

### 10.2.3 DFS vs BFS

| Aspect | DFS | BFS |
|--------|-----|-----|
| **Data Structure** | Stack (recursion) | Queue |
| **Memory** | O(h) - height | O(w) - width |
| **Path Found** | May not be shortest | Shortest (unweighted) |
| **Use When** | Detect cycles, topology | Shortest path, levels |
| **Graph Type** | Deep, sparse | Wide, dense |

**Example Comparison:**
```
Graph:        A
            /   \
           B     C
          / \   / \
         D   E F   G

DFS: A → B → D → E → C → F → G (deep first)
BFS: A → B → C → D → E → F → G (level by level)
```

---

## 10.3 Common Graph Problems

### 10.3.1 Number of Islands

```python
def num_islands(grid):
    """
    LeetCode 200: Number of Islands
    
    Count number of islands (connected 1s) in 2D grid.
    
    Approach: DFS/BFS from each unvisited 1.
    """
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    islands = 0
    
    def dfs(r, c):
        # Boundary check
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
            return
        
        # Mark as visited
        grid[r][c] = '0'
        
        # Visit all 4 directions
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)
    
    # Check each cell
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                islands += 1
                dfs(r, c)  # Mark entire island
    
    return islands

# Time: O(rows × cols)
# Space: O(rows × cols) - recursion stack in worst case

# Example
grid = [
    ["1","1","0","0","0"],
    ["1","1","0","0","0"],
    ["0","0","1","0","0"],
    ["0","0","0","1","1"]
]
print(num_islands(grid))  # 3
```

---

### 10.3.2 Clone Graph

```python
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors else []

def clone_graph(node):
    """
    LeetCode 133: Clone Graph
    
    Deep copy of undirected graph.
    
    Approach: DFS/BFS with hash map for clones.
    """
    if not node:
        return None
    
    # Map: original → clone
    clones = {}
    
    def dfs(original):
        if original in clones:
            return clones[original]
        
        # Create clone
        clone = Node(original.val)
        clones[original] = clone
        
        # Clone neighbors
        for neighbor in original.neighbors:
            clone.neighbors.append(dfs(neighbor))
        
        return clone
    
    return dfs(node)

# Time: O(V + E)
# Space: O(V)
```

---

### 10.3.3 Course Schedule (Cycle Detection)

```python
def can_finish(num_courses, prerequisites):
    """
    LeetCode 207: Course Schedule
    
    Check if can finish all courses given prerequisites.
    
    Example: [[1,0], [0,1]] → False (cycle)
    
    Approach: Detect cycle in directed graph using DFS.
    """
    # Build adjacency list
    graph = [[] for _ in range(num_courses)]
    for course, prereq in prerequisites:
        graph[course].append(prereq)
    
    # States: 0 = unvisited, 1 = visiting, 2 = visited
    state = [0] * num_courses
    
    def has_cycle(course):
        if state[course] == 1:
            return True  # Cycle detected (back edge)
        
        if state[course] == 2:
            return False  # Already checked
        
        # Mark as visiting
        state[course] = 1
        
        # Check prerequisites
        for prereq in graph[course]:
            if has_cycle(prereq):
                return True
        
        # Mark as visited
        state[course] = 2
        return False
    
    # Check each course
    for course in range(num_courses):
        if has_cycle(course):
            return False
    
    return True

# Time: O(V + E)
# Space: O(V + E)

# Example
print(can_finish(2, [[1, 0]]))  # True
print(can_finish(2, [[1, 0], [0, 1]]))  # False (cycle)
```

---

### 10.3.4 Pacific Atlantic Water Flow

```python
def pacific_atlantic(heights):
    """
    LeetCode 417: Pacific Atlantic Water Flow
    
    Find cells where water can flow to both oceans.
    
    Water flows from high to low or equal.
    Pacific: top and left edges
    Atlantic: bottom and right edges
    
    Approach: DFS from both oceans, find intersection.
    """
    if not heights:
        return []
    
    rows, cols = len(heights), len(heights[0])
    pacific = set()
    atlantic = set()
    
    def dfs(r, c, visited, prev_height):
        # Boundary check
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            (r, c) in visited or heights[r][c] < prev_height):
            return
        
        visited.add((r, c))
        
        # Visit neighbors
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            dfs(r + dr, c + dc, visited, heights[r][c])
    
    # DFS from Pacific (top and left)
    for c in range(cols):
        dfs(0, c, pacific, heights[0][c])
    for r in range(rows):
        dfs(r, 0, pacific, heights[r][0])
    
    # DFS from Atlantic (bottom and right)
    for c in range(cols):
        dfs(rows - 1, c, atlantic, heights[rows - 1][c])
    for r in range(rows):
        dfs(r, cols - 1, atlantic, heights[r][cols - 1])
    
    # Find intersection
    return list(pacific & atlantic)

# Time: O(rows × cols)
# Space: O(rows × cols)

# Example
heights = [
    [1,2,2,3,5],
    [3,2,3,4,4],
    [2,4,5,3,1],
    [6,7,1,4,5],
    [5,1,1,2,4]
]
print(pacific_atlantic(heights))
# [[0,4], [1,3], [1,4], [2,2], [3,0], [3,1], [4,0]]
```

---

### 10.3.5 Word Ladder

```python
def ladder_length(begin_word, end_word, word_list):
    """
    LeetCode 127: Word Ladder
    
    Find shortest transformation sequence from begin to end.
    Each step changes one letter, and intermediate words must be in list.
    
    Approach: BFS (shortest path in unweighted graph).
    """
    if end_word not in word_list:
        return 0
    
    word_set = set(word_list)
    queue = deque([(begin_word, 1)])  # (word, steps)
    
    while queue:
        word, steps = queue.popleft()
        
        if word == end_word:
            return steps
        
        # Try all one-letter changes
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + c + word[i+1:]
                
                if next_word in word_set:
                    word_set.remove(next_word)  # Mark visited
                    queue.append((next_word, steps + 1))
    
    return 0

# Time: O(M² × N) where M = word length, N = word list size
# Space: O(N)

# Example
begin_word = "hit"
end_word = "cog"
word_list = ["hot","dot","dog","lot","log","cog"]
print(ladder_length(begin_word, end_word, word_list))  # 5
# hit → hot → dot → dog → cog
```

---

## 10.4 Shortest Path Algorithms

### 10.4.1 BFS for Unweighted Graphs

**Definition**: BFS finds shortest path in unweighted graphs.

```python
def shortest_path_bfs(graph, start, end):
    """
    Find shortest path from start to end using BFS.
    
    Returns path as list of vertices.
    """
    if start == end:
        return [start]
    
    visited = {start}
    queue = deque([(start, [start])])  # (vertex, path)
    
    while queue:
        vertex, path = queue.popleft()
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                new_path = path + [neighbor]
                
                if neighbor == end:
                    return new_path
                
                visited.add(neighbor)
                queue.append((neighbor, new_path))
    
    return []  # No path found

# Time: O(V + E)
# Space: O(V)

# Example
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}
print(shortest_path_bfs(graph, 'A', 'F'))  # ['A', 'C', 'F']
```

---

### 10.4.2 Dijkstra's Algorithm

**Definition**: Finds shortest path in weighted graph with non-negative weights using greedy approach with priority queue.

**Algorithm:**
1. Initialize distances to infinity (except start = 0)
2. Use min heap to process vertices by distance
3. For each vertex, update distances to neighbors
4. Continue until all vertices processed

```python
import heapq

def dijkstra(graph, start):
    """
    Dijkstra's shortest path algorithm.
    
    graph: dict of dict {u: {v: weight}}
    start: starting vertex
    
    Returns: distances from start to all vertices
    """
    # Initialize distances
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0
    
    # Min heap: (distance, vertex)
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current in visited:
            continue
        
        visited.add(current)
        
        # Update distances to neighbors
        for neighbor, weight in graph[current].items():
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

# Time: O((V + E) log V) with binary heap
# Space: O(V)

# Example
graph = {
    'A': {'B': 4, 'C': 2},
    'B': {'C': 1, 'D': 5},
    'C': {'D': 8, 'E': 10},
    'D': {'E': 2},
    'E': {}
}
distances = dijkstra(graph, 'A')
print(distances)
# {'A': 0, 'B': 4, 'C': 2, 'D': 9, 'E': 11}
```

**With Path Reconstruction:**
```python
def dijkstra_with_path(graph, start, end):
    """
    Dijkstra with path reconstruction.
    """
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0
    previous = {vertex: None for vertex in graph}
    
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current == end:
            break
        
        if current in visited:
            continue
        
        visited.add(current)
        
        for neighbor, weight in graph[current].items():
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current
                heapq.heappush(pq, (distance, neighbor))
    
    # Reconstruct path
    path = []
    current = end
    while current:
        path.append(current)
        current = previous[current]
    
    return distances[end], path[::-1]

# Example
distance, path = dijkstra_with_path(graph, 'A', 'E')
print(f"Distance: {distance}, Path: {path}")
# Distance: 11, Path: ['A', 'C', 'D', 'E']
```

---

### 10.4.3 Bellman-Ford Algorithm

**Definition**: Finds shortest path in weighted graph, handles negative weights, detects negative cycles.

**Algorithm:**
1. Initialize distances (start = 0, others = infinity)
2. Relax all edges V-1 times
3. Check for negative cycles (one more iteration)

```python
def bellman_ford(graph, start, num_vertices):
    """
    Bellman-Ford algorithm.
    
    graph: list of edges [(u, v, weight)]
    start: starting vertex
    num_vertices: total number of vertices
    
    Returns: distances, or None if negative cycle exists
    """
    # Initialize distances
    distances = {i: float('inf') for i in range(num_vertices)}
    distances[start] = 0
    
    # Relax edges V-1 times
    for _ in range(num_vertices - 1):
        for u, v, weight in graph:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
    
    # Check for negative cycles
    for u, v, weight in graph:
        if distances[u] != float('inf') and distances[u] + weight < distances[v]:
            return None  # Negative cycle detected
    
    return distances

# Time: O(V × E)
# Space: O(V)

# Example
edges = [
    (0, 1, 4),
    (0, 2, 2),
    (1, 2, 1),
    (1, 3, 5),
    (2, 3, 8),
    (2, 4, 10),
    (3, 4, 2)
]
distances = bellman_ford(edges, 0, 5)
print(distances)
# {0: 0, 1: 4, 2: 2, 3: 9, 4: 11}
```

---

### 10.4.4 Network Delay Time

```python
def network_delay_time(times, n, k):
    """
    LeetCode 743: Network Delay Time
    
    Find time for signal to reach all nodes.
    times[i] = (u, v, w) - signal from u to v takes w time
    
    Approach: Dijkstra's algorithm
    """
    # Build graph
    graph = {i: [] for i in range(1, n + 1)}
    for u, v, w in times:
        graph[u].append((v, w))
    
    # Dijkstra
    distances = {i: float('inf') for i in range(1, n + 1)}
    distances[k] = 0
    pq = [(0, k)]
    
    while pq:
        time, node = heapq.heappop(pq)
        
        if time > distances[node]:
            continue
        
        for neighbor, weight in graph[node]:
            new_time = time + weight
            
            if new_time < distances[neighbor]:
                distances[neighbor] = new_time
                heapq.heappush(pq, (new_time, neighbor))
    
    max_time = max(distances.values())
    return max_time if max_time != float('inf') else -1

# Time: O((V + E) log V)
# Space: O(V + E)

# Example
times = [[2,1,1], [2,3,1], [3,4,1]]
n = 4
k = 2
print(network_delay_time(times, n, k))  # 2
```

---

### 10.4.5 Cheapest Flights Within K Stops

```python
def find_cheapest_price(n, flights, src, dst, k):
    """
    LeetCode 787: Cheapest Flights Within K Stops
    
    Find cheapest price from src to dst with at most k stops.
    
    Approach: Modified Dijkstra or BFS with k constraint
    """
    # Build graph
    graph = {i: [] for i in range(n)}
    for u, v, price in flights:
        graph[u].append((v, price))
    
    # BFS with cost tracking
    # (cost, city, stops)
    queue = deque([(0, src, 0)])
    
    # Track minimum cost to reach each city with given stops
    costs = {i: float('inf') for i in range(n)}
    
    while queue:
        cost, city, stops = queue.popleft()
        
        if city == dst:
            costs[dst] = min(costs[dst], cost)
            continue
        
        if stops > k:
            continue
        
        for neighbor, price in graph[city]:
            new_cost = cost + price
            
            # Only continue if this is better
            if new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                queue.append((new_cost, neighbor, stops + 1))
    
    return costs[dst] if costs[dst] != float('inf') else -1

# Time: O(V + E × K)
# Space: O(V)

# Example
n = 3
flights = [[0,1,100], [1,2,100], [0,2,500]]
src = 0
dst = 2
k = 1
print(find_cheapest_price(n, flights, src, dst, k))  # 200
```

---

## 10.5 Topological Sort

### 10.5.1 Topological Sort Fundamentals

**Definition**: Linear ordering of vertices in a DAG such that for every directed edge (u, v), u comes before v.

**Requirements:**
- Graph must be DAG (no cycles)
- Multiple valid orderings may exist

**Use Cases:**
- Task scheduling
- Build dependencies
- Course prerequisites
- Compiler optimization

**Visual Example:**
```
Graph:    A → B → D
          ↓   ↓
          C → E

Valid orderings:
- A, B, C, D, E
- A, C, B, E, D
- A, B, D, C, E
```

---

### 10.5.2 Topological Sort using DFS

```python
def topological_sort_dfs(graph, num_vertices):
    """
    Topological sort using DFS.
    
    Returns list of vertices in topological order.
    """
    visited = set()
    stack = []
    
    def dfs(vertex):
        visited.add(vertex)
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                dfs(neighbor)
        
        # Add to stack after visiting all descendants
        stack.append(vertex)
    
    # Visit all vertices
    for vertex in range(num_vertices):
        if vertex not in visited:
            dfs(vertex)
    
    # Stack has reverse topological order
    return stack[::-1]

# Time: O(V + E)
# Space: O(V)

# Example
graph = {
    0: [1, 2],
    1: [3],
    2: [3],
    3: [4],
    4: []
}
print(topological_sort_dfs(graph, 5))  # [0, 2, 1, 3, 4] (one valid order)
```

---

### 10.5.3 Kahn's Algorithm (BFS)

```python
def topological_sort_bfs(graph, num_vertices):
    """
    Kahn's algorithm for topological sort.
    
    Uses in-degree and BFS.
    """
    # Calculate in-degree
    in_degree = [0] * num_vertices
    for vertex in graph:
        for neighbor in graph[vertex]:
            in_degree[neighbor] += 1
    
    # Queue with vertices having in-degree 0
    queue = deque([v for v in range(num_vertices) if in_degree[v] == 0])
    result = []
    
    while queue:
        vertex = queue.popleft()
        result.append(vertex)
        
        # Reduce in-degree of neighbors
        for neighbor in graph[vertex]:
            in_degree[neighbor] -= 1
            
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Check if all vertices processed (no cycle)
    if len(result) != num_vertices:
        return []  # Cycle detected
    
    return result

# Time: O(V + E)
# Space: O(V)

# Example
graph = {
    0: [1, 2],
    1: [3],
    2: [3],
    3: [4],
    4: []
}
print(topological_sort_bfs(graph, 5))  # [0, 1, 2, 3, 4] (one valid order)
```

---

### 10.5.4 Course Schedule II

```python
def find_order(num_courses, prerequisites):
    """
    LeetCode 210: Course Schedule II
    
    Return ordering of courses to take given prerequisites.
    
    Approach: Topological sort (Kahn's algorithm)
    """
    # Build graph and calculate in-degree
    graph = [[] for _ in range(num_courses)]
    in_degree = [0] * num_courses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    # Start with courses having no prerequisites
    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    order = []
    
    while queue:
        course = queue.popleft()
        order.append(course)
        
        # Process dependents
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            
            if in_degree[next_course] == 0:
                queue.append(next_course)
    
    # Check if all courses can be taken
    return order if len(order) == num_courses else []

# Time: O(V + E)
# Space: O(V + E)

# Example
print(find_order(4, [[1,0], [2,0], [3,1], [3,2]]))
# [0, 1, 2, 3] or [0, 2, 1, 3]
```

---

## 10.6 Union-Find (Disjoint Set)

### 10.6.1 Union-Find Fundamentals

**Definition**: Data structure to track disjoint sets and support union and find operations efficiently.

**Operations:**
- **find(x)**: Find which set x belongs to
- **union(x, y)**: Merge sets containing x and y
- **connected(x, y)**: Check if x and y in same set

**Use Cases:**
- Detect cycles in undirected graph
- Find connected components
- Kruskal's MST algorithm
- Network connectivity

---

### 10.6.2 Union-Find Implementation

```python
class UnionFind:
    """
    Union-Find with path compression and union by rank.
    """
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size
    
    def find(self, x):
        """
        Find root of x with path compression.
        
        Path compression: make all nodes point directly to root.
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        """
        Union sets containing x and y.
        
        Union by rank: attach smaller tree to larger tree.
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already in same set
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        return True
    
    def connected(self, x, y):
        """Check if x and y are in same set"""
        return self.find(x) == self.find(y)

# Time: O(α(n)) ≈ O(1) amortized (α = inverse Ackermann)
# Space: O(n)

# Example
uf = UnionFind(5)
uf.union(0, 1)
uf.union(1, 2)
print(uf.connected(0, 2))  # True
print(uf.connected(0, 3))  # False
```

---

### 10.6.3 Number of Connected Components

```python
def count_components(n, edges):
    """
    LeetCode 323: Number of Connected Components in Undirected Graph
    
    Count number of connected components.
    
    Approach: Union-Find
    """
    uf = UnionFind(n)
    
    # Union all edges
    for u, v in edges:
        uf.union(u, v)
    
    # Count unique roots
    return len(set(uf.find(i) for i in range(n)))

# Time: O(E × α(V))
# Space: O(V)

# Example
n = 5
edges = [[0,1], [1,2], [3,4]]
print(count_components(n, edges))  # 2
```

---

### 10.6.4 Graph Valid Tree

```python
def valid_tree(n, edges):
    """
    LeetCode 261: Graph Valid Tree
    
    Check if edges form a valid tree.
    
    Tree properties:
    1. n-1 edges
    2. No cycles
    3. Connected
    
    Approach: Union-Find to detect cycle + check edge count
    """
    # Tree must have exactly n-1 edges
    if len(edges) != n - 1:
        return False
    
    uf = UnionFind(n)
    
    # Check for cycles
    for u, v in edges:
        if not uf.union(u, v):
            return False  # Cycle detected
    
    return True

# Time: O(E × α(V))
# Space: O(V)

# Example
print(valid_tree(5, [[0,1], [0,2], [0,3], [1,4]]))  # True
print(valid_tree(5, [[0,1], [1,2], [2,3], [1,3], [1,4]]))  # False (cycle)
```

---

## Practice Questions

### Fill in the Gaps

1. BFS uses a ________ while DFS uses a ________.
2. The time complexity of BFS and DFS is ________.
3. Dijkstra's algorithm doesn't work with ________ edge weights.
4. Topological sort only works on ________ graphs.
5. Union-Find with path compression has ________ amortized time.

### True or False

1. BFS always finds the shortest path. **[T/F]**
2. DFS uses less memory than BFS for wide graphs. **[T/F]**
3. Dijkstra's algorithm uses a priority queue. **[T/F]**
4. A DAG can have multiple valid topological orderings. **[T/F]**
5. Union-Find can detect cycles in directed graphs. **[T/F]**

### Multiple Choice

1. Best algorithm for shortest path with negative weights?
   - A) BFS
   - B) Dijkstra
   - C) Bellman-Ford
   - D) DFS

2. Time complexity of Dijkstra with binary heap?
   - A) O(V)
   - B) O(E)
   - C) O((V+E) log V)
   - D) O(V²)

3. Which traversal visits nodes level by level?
   - A) DFS
   - B) BFS
   - C) Topological sort
   - D) Union-Find

### Code Challenge

```python
def surrounded_regions(board):
    """
    LeetCode 130: Surrounded Regions
    
    Capture all 'O' regions surrounded by 'X'.
    Regions on border are not captured.
    
    Example:
    X X X X    X X X X
    X O O X -> X X X X
    X X O X    X X X X
    X O X X    X O X X
    
    Use DFS/BFS from border 'O' cells.
    """
    # Your code here
    pass
```

---

## Answers

<details>
<summary><strong>View Answers</strong></summary>

### Fill in the Gaps

1. **queue, stack** (or recursion)
2. **O(V + E)**
3. **negative**
4. **directed acyclic (DAG)**
5. **O(α(n))** or **nearly O(1)**

### True or False

1. **False** - Only for unweighted graphs
2. **True** - DFS space is O(height), BFS is O(width)
3. **True** - Min heap/priority queue
4. **True** - Multiple valid orderings possible
5. **False** - Works for undirected only (need DFS for directed)

### Multiple Choice

1. **C** - Bellman-Ford handles negative weights
2. **C** - Each vertex/edge processed with heap operations
3. **B** - BFS is level-order traversal

### Code Challenge Answer

```python
def surrounded_regions(board):
    if not board:
        return
    
    rows, cols = len(board), len(board[0])
    
    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != 'O':
            return
        
        # Mark as safe (border-connected)
        board[r][c] = 'S'
        
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)
    
    # Mark border-connected 'O' as safe
    for r in range(rows):
        dfs(r, 0)
        dfs(r, cols - 1)
    
    for c in range(cols):
        dfs(0, c)
        dfs(rows - 1, c)
    
    # Capture surrounded regions and restore safe ones
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == 'O':
                board[r][c] = 'X'  # Capture
            elif board[r][c] == 'S':
                board[r][c] = 'O'  # Restore

# Time: O(rows × cols)
# Space: O(rows × cols) - recursion stack
```

</details>

---

## LeetCode Problems (NeetCode.io)

### Graphs - Medium 🟨
- 133. Clone Graph (IMPORTANT)
- 200. Number of Islands (VERY IMPORTANT)
- 207. Course Schedule (IMPORTANT - Cycle Detection)
- 210. Course Schedule II (Topological Sort)
- 261. Graph Valid Tree
- 323. Number of Connected Components
- 417. Pacific Atlantic Water Flow
- 547. Number of Provinces
- 695. Max Area of Island
- 743. Network Delay Time (Dijkstra)
- 787. Cheapest Flights Within K Stops
- 994. Rotting Oranges

### Graphs - Hard 🔴
- 127. Word Ladder (IMPORTANT)
- 130. Surrounded Regions
- 269. Alien Dictionary
- 684. Redundant Connection
- 778. Swim in Rising Water

---

## Summary

**Traversal:**
- **DFS**: Stack/recursion, goes deep, O(V+E)
- **BFS**: Queue, goes wide, shortest path in unweighted

**Shortest Path:**
- **BFS**: Unweighted graphs, O(V+E)
- **Dijkstra**: Non-negative weights, O((V+E) log V)
- **Bellman-Ford**: Handles negative weights, O(VE)

**Other:**
- **Topological Sort**: DFS or Kahn's, O(V+E)
- **Union-Find**: Disjoint sets, O(α(n)) ≈ O(1)

---

*Continue to: [11. Dynamic Programming →](11-dynamic-programming.md)*