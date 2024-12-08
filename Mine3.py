import logging
import time
from collections import defaultdict
import copy
'''这是gpt给的进度日志版本，但不是我想要的日志，只是翻译我的pprint'''
# Configure logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Edge():
    def __init__(self, eId, eLb, frmId, frmLb, toLb, toId):
        self.eId = eId
        self.eLb = eLb
        self.frmId = frmId
        self.frmLb = frmLb
        self.toId = toId
        self.toLb = toLb

class Vertex():
    def __init__(self, vId, vLb):
        self.vId = vId
        self.vLb = vLb
        self.edges = dict()  # Stores all edges originating from the current vertex {eId:edge}

class Graph():
    def __init__(self):
        self.vertices = dict()  # Stores all vertices
        self.edges = dict()  # Stores all edges

    def addEdge(self, edge):
        logging.info(f"Adding edge {edge.eId} from vertex {edge.frmId} to vertex {edge.toId}")
        self.edges[edge.eId] = edge
        if edge.frmId not in self.vertices:
            self.vertices[edge.frmId] = Vertex(edge.frmId, edge.frmLb)
        self.vertices[edge.frmId].edges[edge.eId] = edge
        if edge.toId not in self.vertices:
            self.vertices[edge.toId] = Vertex(edge.toId, edge.toLb)

    def addVert(self, vertex):
        logging.info(f"Adding vertex {vertex.vId} with label {vertex.vLb}")
        self.vertices[vertex.vId] = vertex

    def inQueue(self):
        queue = Queue()
        for edge in self.edges.values():
            queue.inq(PDfsCode([edge]))
        return queue

class PDfsCode(list):  # Defines: projected-DfsCode, storing Edge
    def __init__(self, edges=None):
        super().__init__()
        if edges is not None:
            self.extend(edges)

    def ToGraph(self):
        # Construct graph from DfsCode
        g = Graph()
        for edge in self:
            g.addEdge(edge)
        return g

    def ToDfsCode(self):  # Extract DfsCode information from PDfsCode
        dfs_code = []
        for edge in self:
            eveLb = (edge.frmLb, edge.eLb, edge.toLb)
            dfs_code.append(eveLb)
        return tuple(dfs_code)

    def intigralKey(self):
        temp = []
        for edge in self:
            temp.append((edge.frmId, edge.toId, edge.eId, (edge.frmLb, edge.eLb, edge.toLb)))
        return tuple(temp)

class PDfsCoLi(list):  # Stores a list of PDfsCode for easy iteration and constructing higher-order projects
    def __init__(self, p_codes=None):
        super().__init__()
        if p_codes is not None:
            self.extend(p_codes)

class Projected(dict):  # {"DfsCode": PDfsCoLi}
    def __init__(self):
        super().__init__()

    def add_PdfsCode(self, p_dfs_code, dfs_code):
        logging.info(f"Adding PDfsCode to projected with key {dfs_code}")
        if dfs_code in self:
            self[dfs_code].append(p_dfs_code)
        else:
            self[dfs_code] = PDfsCoLi([p_dfs_code])

class SubGraphs(dict):  # Subgraph DfsCode and their support
    def __init__(self):
        super().__init__()

    def add(self, dfs_code, support):
        logging.info(f"Adding subgraph with DfsCode {dfs_code} and support {support}")
        self[dfs_code] = support

class Visited():
    def __init__(self):
        self.vertices = set()  # id
        self.edges = set()  # id

    def GraphIn(self, g):  # Write graph into Visited
        for eId, edge in g.edges.items():
            self.edges.add(eId)
            self.vertices.add(edge.frmId)
            self.vertices.add(edge.toId)

    def EdgeIn(self, e):  # Write edge into Visited
        self.edges.add(e.eId)
        self.vertices.add(e.frmId)
        self.vertices.add(e.toId)

class Queue(list):  # Stores PDfsCode
    def __init__(self):
        super().__init__()

    def inq(self, a):
        self.append(a)

    def deq(self):
        return self.pop(0)

    def ToDfsQueue(self):
        tem_queue = []
        for p_dfs_code in self:
            a = p_dfs_code.ToDfsCode()
            tem_queue.append(a)
        return tem_queue

# Reverse expansion, if not possible, perform forward expansion
def rmpath(G, p_dfs_code, visited):
    logging.info(f"Attempting reverse expansion for PDfsCode with edges {[(e.eId, e.frmId, e.toId) for e in p_dfs_code]}")
    g = p_dfs_code.ToGraph()
    visited.GraphIn(g)
    for eId, edge in G.vertices[p_dfs_code[-1].toId].edges.items():
        if edge.toId in visited.vertices and eId not in visited.edges:
            visited.EdgeIn(edge)
            logging.info(f"Found reverse edge {edge.eId} from {edge.frmId} to {edge.toId}")
            return edge
    # No reverse edge, look for forward edge
    for eId, edge in G.vertices[p_dfs_code[-1].toId].edges.items():
        if edge.toId not in visited.vertices:
            visited.EdgeIn(edge)
            logging.info(f"Found forward edge {edge.eId} from {edge.frmId} to {edge.toId}")
            return edge
    logging.info("No more possible expansions")  # Confirm returning None
    return None

# Auxiliary function: Find Min_DfsCode for a graph, starting by putting each edge in the queue
def findMin(g, queue):
    logging.info("Finding minimum DfsCode for the current subgraph")
    h_ = len(queue)
    for i in range(h_):
        h = i
        queue_i = copy.copy(queue[i])
        e = rmpath(g, queue[i], visited=Visited())
        if e is not None:
            break  # Indicates expandable
    if e is None:  # Not expandable
        tem_queue = queue.ToDfsQueue()
        logging.info(f"Minimum path for current subgraph is {min(tem_queue)}")
        return min(tem_queue)
    else:  # At least one is expandable
        for i in range(h + 1):
            queue.deq()  # Next level exists, dequeue previous level
        queue_i.append(e)
        queue.inq(queue_i)
        visited = Visited()
        for i in range(h_ - h - 1):
            top = queue.deq()
            e = rmpath(g, top, visited)  # Give an expansion edge
            if e is not None:
                top.append(e)
                queue.inq(top)
        return findMin(g, queue)

# Mine projected, input is projected, recursive on pointer, modifying frequent sets in place
def SubMining(minedSubGraphs, projected, G, supp_thres):
    if not projected:
        logging.info("Mining complete")
        return

    total_iterations = len(projected)
    start_time = time.time()
    for idx, (DfsCode, PDfsCoLi) in enumerate(projected.items()):
        progress = (idx + 1) / total_iterations * 100
        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / (idx + 1)) * total_iterations if idx > 0 else 0
        estimated_remaining_time = estimated_total_time - elapsed_time
        logging.info(f"Progress: {progress:.2f}% - Estimated time remaining: {estimated_remaining_time:.2f} seconds")

        logging.info(f"Mining DfsCode {DfsCode}")
        # Construct new projected
        projected_new = Projected()

        for p_dfs_code in PDfsCoLi:
            logging.info(f"Iterating PDfsCode with edges {[ (i.eId, i.frmId, i.toId) for i in p_dfs_code]} ")
            visited = Visited()
            g = p_dfs_code.ToGraph()
            visited.GraphIn(g)
            while True:
                e = rmpath(G, p_dfs_code, visited)
                if e is None:
                    break
                logging.info(f"Expanded edge: {e.eId} from {e.frmId} to {e.toId}")
                PCoCp = copy.copy(p_dfs_code)
                PCoCp.append(e)
                DfsCoCp = PCoCp.ToDfsCode()
                projected_new.add_PdfsCode(PCoCp, DfsCoCp)

        # Now filter out those that exceed the frequency threshold
        logging.info("Filtering subgraphs based on support threshold")
        projected_filt = Projected()
        for key, val in projected_new.items():
            g = val[0].ToGraph()
            queue = g.inQueue()
            dfs_min = findMin(g, queue)
            if dfs_min == key and len(val) >= supp_thres:
                logging.info(f"Path {key} is the minimum path and meets support threshold")
                projected_filt[key] = val
                inti_key = val[0].intigralKey()
                minedSubGraphs[inti_key] = len(val)
                logging.info(f"Mined subgraph {inti_key} with support {len(val)}")
            else:
                logging.info("Current path is not the minimum path")

        # Continue mining in the minimal frequent set
        SubMining(minedSubGraphs, projected_filt, G, supp_thres)

def ReadGraph(filepath):
    logging.info(f"Reading graph from {filepath}")
    G = Graph()  # Initialize a Graph object
    with open(filepath, 'r') as file:
        lines = file.readlines()
        current_graph = G
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':  # Read vertex info
                vId = int(parts[1])
                vLb = parts[2]  # Vertex label
                vertex = Vertex(vId, vLb)
                current_graph.addVert(vertex)
            elif parts[0] == 'e':  # Read edge info
                frmId = int(parts[1])
                toId = int(parts[2])
                eId = len(current_graph.edges) + 1  # Auto-generate edge ID
                eLb = "-".join(parts[3:])  # Edge label
                edge = Edge(eId, eLb, frmId, current_graph.vertices[frmId].vLb,
                            current_graph.vertices[toId].vLb, toId)
                current_graph.addEdge(edge)
    return G

# Main function
if __name__ == "__main__":
    minedSubGraphs = SubGraphs()
    projected = Projected()
    G = ReadGraph("D:\软件\python及系列软件\环境测试\gSpan-master3\merged_graph2.data")  # 读取图
    supp_thres = 1

    # Insert each edge of G into projected
    logging.info("Initializing projected with individual edges")
    for eId, edge in G.edges.items():
        p_dfs_code = PDfsCode([edge])
        a = p_dfs_code.ToDfsCode()
        projected.add_PdfsCode(p_dfs_code, a)
    # Filter and mine subgraphs
    logging.info("Filtering and mining initial subgraphs")
    projected_filt = Projected()
    for key, val in projected.items():
        if len(val) >= supp_thres:  # Frequent subgraph
            projected_filt[key] = val
            inti_key = val[0].intigralKey()
            minedSubGraphs[inti_key] = len(val)

    SubMining(minedSubGraphs, projected_filt, G, supp_thres)
