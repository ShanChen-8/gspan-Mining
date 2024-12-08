from collections import defaultdict

class Graph:
    def __init__(self):
        self.edges = defaultdict(list)
        self.vertices = set()

    def add_edge(self, src, dest):
        self.edges[src].append(dest)
        self.edges[dest].append(src)
        self.vertices.add(src)
        self.vertices.add(dest)

class DFSCode:
    def __init__(self, src, dest, label):
        self.src = src
        self.dest = dest
        self.label = label

    def __repr__(self):
        return f"({self.src}, {self.dest}, {self.label})"

class GSpan:
    def __init__(self, graphs, min_support):
        self.graphs = graphs
        self.min_support = min_support
        self.frequent_subgraphs = []

    def run(self):
        for graph_id, graph in enumerate(self.graphs):
            self.dfs_recursive([], graph_id, graph, set())

    def dfs_recursive(self, dfs_code, graph_id, graph, visited):
        if len(dfs_code) > 0:
            support = self.compute_support(dfs_code)
            if support >= self.min_support:
                self.frequent_subgraphs.append(dfs_code)

        for src in graph.vertices:
            if src not in visited:
                for dest in graph.edges[src]:
                    if dest not in visited:
                        dfs_code.append(DFSCode(src, dest, "label"))
                        visited.add(src)
                        visited.add(dest)
                        self.dfs_recursive(dfs_code, graph_id, graph, visited)
                        dfs_code.pop()
                        visited.remove(src)
                        visited.remove(dest)

    def compute_support(self, dfs_code):
        support_count = 0
        for graph in self.graphs:
            if self.is_subgraph(graph, dfs_code):
                support_count += 1
        return support_count

    def is_subgraph(self, graph, dfs_code):
        visited = set()
        return self.match_dfs_code(graph, dfs_code, 0, visited)

    def match_dfs_code(self, graph, dfs_code, index, visited):
        if index == len(dfs_code):
            return True

        current_edge = dfs_code[index]
        if current_edge.src in graph.vertices and current_edge.dest in graph.edges[current_edge.src]:
            if (current_edge.src, current_edge.dest) not in visited and (current_edge.dest, current_edge.src) not in visited:
                visited.add((current_edge.src, current_edge.dest))
                if self.match_dfs_code(graph, dfs_code, index + 1, visited):
                    return True
                visited.remove((current_edge.src, current_edge.dest))
        return False

# Example usage
graphs = []
graph1 = Graph()
graph1.add_edge(1, 2)
graph1.add_edge(2, 3)
graphs.append(graph1)

graph2 = Graph()
graph2.add_edge(1, 2)
graph2.add_edge(2, 4)
graphs.append(graph2)

gspan = GSpan(graphs, min_support=1)
gspan.run()
print("Frequent subgraphs:")
for subgraph in gspan.frequent_subgraphs:
    print(subgraph)
