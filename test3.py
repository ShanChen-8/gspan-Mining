from collections import defaultdict


def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start, end=' ')
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)


# def main():
    # 创建图的邻接表表示
graph = defaultdict(list)
graph[0] = [1, 2]
graph[1] = [0, 3, 4]
graph[2] = [0]
graph[3] = [1]
graph[4] = [1, 5]
graph[5] = [4]

# 从节点 0 开始进行 DFS 遍历
print("DFS traversal starting from node 0:")
dfs(graph, 0)


# if __name__ == "__main__":
#     main()
