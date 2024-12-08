import networkx as nx
from collections import defaultdict

class GSpan:
    def __init__(self, graphs, min_support):
        self.graphs = graphs
        self.min_support = min_support
        self.frequent_subgraphs = []

    def run(self):
        # Generate frequent subgraphs (This is a simplified representation)
        frequent_edges = self.get_frequent_edges()
        for edge in frequent_edges:
            subgraph = nx.Graph()
            subgraph.add_edge(edge[0], edge[1], label=edge[2])
            self.frequent_subgraphs.append(subgraph)

    def get_frequent_edges(self):
        edge_counts = defaultdict(int)
        for graph in self.graphs:
            for u, v, data in graph.edges(data=True):
                edge_counts[(u, v, data['label'])] += 1
        return [e for e, count in edge_counts.items() if count >= self.min_support]

    def print_frequent_subgraphs(self):
        for i, subgraph in enumerate(self.frequent_subgraphs):
            print(f"Frequent Subgraph #{i + 1}:")
            print(subgraph.edges(data=True))


def parse_graph_data(data):
    graphs = []
    current_graph = None

    for line in data.split("\n"):
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == 't':
            if tokens[1] == '#':
                if current_graph is not None:
                    graphs.append(current_graph)
                current_graph = nx.Graph()
        elif tokens[0] == 'v':
            node_id = int(tokens[1])
            label = tokens[2]
            current_graph.add_node(node_id, label=label)
        elif tokens[0] == 'e':
            node1 = int(tokens[1])
            node2 = int(tokens[2])
            label = tokens[3]
            current_graph.add_edge(node1, node2, label=label)

    if current_graph is not None:
        graphs.append(current_graph)
    return graphs


data = '''
t # 0
v 9 Jobs
v 8 John
v 7 Mike
v 6 Jobs
v 5 John
v 4 Mike
v 3 Jobs
v 2 John
v 1 Mike
v 0 Jobs
e 7 8 0 strategy_name-4 buscode3
e 8 9 0 strategy_name-4 buscode3
e 9 7 0 strategy_name-4 buscode3
e 4 5 0 strategy_name-4 buscode3
e 5 6 0 strategy_name-4 buscode3
e 6 4 0 strategy_name-4 buscode3
e 1 2 0 strategy_name-4 buscode3
e 2 3 0 strategy_name-4 buscode3
e 3 1 0 strategy_name-4 buscode3
e 1 2 0 strategy_name-4 buscode3
e 2 3 0 strategy_name-4 buscode3
e 3 1 0 strategy_name-4 buscode3
t # -1
'''

graphs = parse_graph_data(data)
min_support = 2
gspan = GSpan(graphs, min_support)
gspan.run()
gspan.print_frequent_subgraphs()
