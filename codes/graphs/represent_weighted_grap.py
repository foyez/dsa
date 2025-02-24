def represent_weighted_graph(n, m, c):
    adj_matrix = [[0] * (n + 1) for row in range(n + 1)]
    # space: O(n^2)

    for n1, n2, w in c:
        adj_matrix[n1][n2] = w
        adj_matrix[n2][n1] = w # only for undirected graph

    print(adj_matrix)

    adj_list = [[] for row in range(n+1)]
    # space: O(2E) - for undirected graph
    # space: O(E) - for directed graph

    for n1, n2, w in c:
        adj_list[n1].append((n2, w),)
        adj_list[n2].append((n1, w),) # only for undirected graph

    print(adj_list)


nodes = 5
edges = 6
connections = [[1,2, 2], [1,3, 3], [3,4, 4],[2,4, 1],[2,5, 6],[4,5, 3]]
# 1 ---- 2-----|
# |      |     5
# 3 ---- 4-----|
represent_weighted_graph(nodes, edges, connections)