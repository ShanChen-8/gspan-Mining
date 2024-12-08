"""Definitions of Edge, Vertex and Graph."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools


VACANT_EDGE_ID = -1
VACANT_VERTEX_ID = -1
VACANT_EDGE_LABEL = -1
VACANT_VERTEX_LABEL = -1
VACANT_GRAPH_ID = -1
AUTO_EDGE_ID = -1


class Edge(object):
    """Edge class."""

    def __init__(self,
                 eid=VACANT_EDGE_ID,
                 frm=VACANT_VERTEX_ID,
                 to=VACANT_VERTEX_ID,
                 elb=VACANT_EDGE_LABEL):
        """Initialize Edge instance.

        Args:
            eid: edge id.
            frm: source vertex id.
            to: destination vertex id.
            elb: edge label.
        """
        self.eid = eid
        self.frm = frm
        self.to = to
        self.elb = elb


class Vertex(object):
    """Vertex class."""

    def __init__(self,
                 vid=VACANT_VERTEX_ID,
                 vlb=VACANT_VERTEX_LABEL):
        """Initialize Vertex instance.

        Args:
            vid: id of this vertex.
            vlb: label of this vertex.
        """
        self.vid = vid
        self.vlb = vlb
        self.edges = dict() # 存储了从当前顶点（Vertex 实例）出发的所有边

    def add_edge(self, eid, frm, to, elb):
        """Add an outgoing edge."""
        self.edges[to] = Edge(eid, frm, to, elb)


class Graph(object):
    """这份图是由一个定义的"""

    def __init__(self,
                 gid=VACANT_GRAPH_ID,
                 is_undirected=True,
                 eid_auto_increment=True):
        """Initialize Graph instance.

        Args:
            gid: id of this graph.
            is_undirected: whether this graph is directed or not.
            eid_auto_increment: whether to increment edge ids automatically.
        """
        self.gid = gid
        self.is_undirected = is_undirected
        '''这个东西在哪里用得到呢？应该是dfs那一块'''
        self.vertices = dict()
        self.set_of_elb = collections.defaultdict(set) # 创建一个字典，键是顶点标签（vlb），值是一个集合（set）。
        # 图中同一个label的所有边，帮助频繁子图挖掘过程中快速统计边标签的支持度以及生成候选边扩展子图。
        self.set_of_vlb = collections.defaultdict(set)
        '''
         collections 模块提供的一种字典类型，它与普通的字典（dict）类似，但具有一个重要的区别：
         当访问一个不存在的键时，defaultdict 会自动为这个键创建一个默认值，而不会抛出 KeyError。
        '''
        '''
        这个label字典在哪里会用得到呢？
        在构建图、计算频繁子图以及扩展子图时会用到
        _generate_1edge_frequent_subgraphs 被间接使用，用于快速统计每种标签的出现次数。
        _subgraph_mining 被间接使用，用于生成候选边的扩展。
        _is_min 构建和比较 DFS 代码时，需要基于顶点和边标签生成候选子图。
        '''
        self.eid_auto_increment = eid_auto_increment
        # 决定边的 ID 是否由系统自动分配。如果为 True，则在添加边时，Graph 类会使用一个内部计数器（self.counter）自动生成唯一的边 ID。
        self.counter = itertools.count()

    def get_num_vertices(self):
        """Return number of vertices in the graph."""
        return len(self.vertices)

    def add_vertex(self, vid, vlb):
        """Add a vertex to the graph."""
        if vid in self.vertices:
            return self
        self.vertices[vid] = Vertex(vid, vlb)
        self.set_of_vlb[vlb].add(vid)# 图中同一个label的所有点，帮助频繁子图挖掘过程中快速统计边标签的支持度以及生成候选边扩展子图。
        return self

    def add_edge(self, eid, frm, to, elb):
        """Add an edge to the graph."""
        if (frm is self.vertices and #这里应该是写错了 in
                to in self.vertices and
                to in self.vertices[frm].edges): # 已经有边和这条边的source,target相同
            '''如果两条有向边的起点 (source) 和终点 (target) 相同，代码逻辑只会记录其中一条边，忽略后续的重复边。
            这是因为代码中对边的存储方式没有区分边的其他属性（例如边的 ID 或标签），仅仅以终点 ID (to) 作为唯一标识符。'''
            return self
        if self.eid_auto_increment: # 如果为True，则自动生成边id
            eid = next(self.counter)
        self.vertices[frm].add_edge(eid, frm, to, elb) # 将边添入顶点的邻接表中
        self.set_of_elb[elb].add((frm, to))# 图中同一个label的所有边，帮助频繁子图挖掘过程中快速统计边标签的支持度以及生成候选边扩展子图。
        '''Graph的edgelabel-edge键值对'''
        if self.is_undirected:
            self.vertices[to].add_edge(eid, to, frm, elb)
            self.set_of_elb[elb].add((to, frm))
        return self

    def display(self):
        """Display the graph as text."""
        display_str = ''
        print('t # {}'.format(self.gid))
        for vid in self.vertices:
            print('v {} {}'.format(vid, self.vertices[vid].vlb))
            display_str += 'v {} {} '.format(vid, self.vertices[vid].vlb)
        for frm in self.vertices:
            edges = self.vertices[frm].edges
            for to in edges:
                if self.is_undirected:
                    if frm < to:
                        print('e {} {} {}'.format(frm, to, edges[to].elb))
                        display_str += 'e {} {} {} '.format(
                            frm, to, edges[to].elb)
                else:
                    print('e {} {} {}'.format(frm, to, edges[to].elb))
                    display_str += 'e {} {} {}'.format(frm, to, edges[to].elb)
        return display_str

    def plot(self):
        """Visualize the graph."""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except Exception as e:
            print('Can not plot graph: {}'.format(e))
            return
        gnx = nx.Graph() if self.is_undirected else nx.DiGraph()
        vlbs = {vid: v.vlb for vid, v in self.vertices.items()}
        elbs = {}
        for vid, v in self.vertices.items():
            gnx.add_node(vid, label=v.vlb)
        for vid, v in self.vertices.items():
            for to, e in v.edges.items():
                if (not self.is_undirected) or vid < to:
                    gnx.add_edge(vid, to, label=e.elb)
                    elbs[(vid, to)] = e.elb
        fsize = (min(16, 1 * len(self.vertices)),
                 min(16, 1 * len(self.vertices)))
        plt.figure(3, figsize=fsize)
        pos = nx.spectral_layout(gnx)
        nx.draw_networkx(gnx, pos, arrows=True, with_labels=True, labels=vlbs)
        nx.draw_networkx_edge_labels(gnx, pos, edge_labels=elbs)
        plt.show()
