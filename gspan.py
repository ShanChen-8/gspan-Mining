"""Implementation of gSpan."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import copy
import itertools
import time

from .graph import AUTO_EDGE_ID
from .graph import Graph
from .graph import VACANT_GRAPH_ID
from .graph import VACANT_VERTEX_LABEL

import pandas as pd


def record_timestamp(func):
    """Record timestamp before and after call of `func`."""
    def deco(self):
        self.timestamps[func.__name__ + '_in'] = time.time()
        func(self)
        self.timestamps[func.__name__ + '_out'] = time.time()
    return deco


class DFSedge(object):
    '''
    定义了两个DFSedge是否相等，不等，和字符串表达函数，
    DFSedge仅仅是一个词典序的一个标签
    我们知道两个子图同构当且仅当最小序DFSedge_list相同
    这个函数给出但：两条边是否相同
    这个函数给出了答案：两个子图是否同构
    那么判定两个子图是否同构的前提在于，转化为最小DFSedge并比较
    在gspan算法中，计算支持度的时候才需要比较图同构，具体而言：
        在某个k-1频繁模式上生成所有k频繁模式(elb,vlb,elb)形成的DFS
        如果有两个k-频繁图的投影图相同，他们一定具有相同的DFSedge，故而：
            先判断该k-频繁模式是否是最小序列，再统计该k-频繁模式的支持度
    '''

    def __init__(self, frm, to, vevlb):
        """Initialize DFSedge instance."""
        self.frm = frm
        self.to = to
        self.vevlb = vevlb # 边的标签或权重(vlb1, elb, vlb2)

    def __eq__(self, other):
        """Check equivalence of DFSedge."""
        '''定义了两个 DFSedge 实例是否相等的条件，即它们的 frm、to 和 vevlb 属性都相等。'''
        return (self.frm == other.frm and
                self.to == other.to and
                self.vevlb == other.vevlb)

    def __ne__(self, other):
        """Check if not equal."""
        return not self.__eq__(other)

    def __repr__(self):
        """Represent DFScode in string way."""
        return '(frm={}, to={}, vevlb={})'.format(
            self.frm, self.to, self.vevlb
        )

# 从pushback函数来看，传入的应该是一个列表，或者直接实例化为一个列表
class DFScode(list):
    """
    DFScode is a list of DFSedge.
    定义了两个DFScode序列是否相等，不等，和字符串表达，
    定义了从图得到DFScode，DFScode得到图，建立最右路径，得到DFScode的顶点数
    我们知道两个子图同构当且仅当最小序DFSedge_list相同
    这个函数给出了答案：两个子图是否同构
    那么判定两个子图是否同构的前提在于，转化为最小DFSedge并比较
    在gspan算法中，计算支持度的时候才需要比较图同构，具体而言：
        在某个k-1频繁模式上生成所有k频繁模式(elb,vlb,elb)形成的DFS
        如果有两个k-频繁图的投影图相同，他们一定具有相同的DFSedge，故而：
            先判断该k-频繁模式是否是最小序列，再统计该k-频繁模式的支持度
    从图建立DFScode竟然没有建立，我本以为这个是核心，gspan过程是顺着DFS拓展图，所以有从DFScode到图，没有从图到DFScode
    """
    '''
    DFScode 在定义时直接继承了 list。
    这意味着 DFScode 是 list 的一个子类。
    '''
    def __init__(self):
        """Initialize DFScode."""
        self.rmpath = list() # 这一句可能是多余的
    '''判断传递给DFScode的列表与其它列表是否相等'''
    def __eq__(self, other):
        """Check equivalence of DFScode."""
        la, lb = len(self), len(other)
        '''这里的 len(self) 是调用了 list 的 __len__ 方法，它返回的是 self（即 DFScode 实例）中存储的
         DFSedge 对象的数量，而不是 rmpath 的长度。
         len(self)等同于len(某个列表)，列表中元素是DFSedge'''
        if la != lb:
            return False
        for i in range(la):
            if self[i] != other[i]:
                return False
        return True

    def __ne__(self, other):
        """Check if not equal."""
        return not self.__eq__(other)

    def __repr__(self):
        """Represent DFScode in string way."""
        return ''.join(['[', ','.join(
            [str(dfsedge) for dfsedge in self]), ']']
        )
    # 把一个DFSedge 附在DFScode列表后面
    def push_back(self, frm, to, vevlb):
        """Update DFScode by adding one edge."""
        self.append(DFSedge(frm, to, vevlb))
        return self
    # 如果未明确指定图ID，可以使用 VACANT_GRAPH_ID 作为默认值
    def to_graph(self, gid=VACANT_GRAPH_ID, is_undirected=True):
        """Construct a graph according to the dfs code."""
        g = Graph(gid,
                  is_undirected=is_undirected,
                  eid_auto_increment=True)
        for dfsedge in self:
            frm, to, (vlb1, elb, vlb2) = dfsedge.frm, dfsedge.to, dfsedge.vevlb
            if vlb1 != VACANT_VERTEX_LABEL:
                g.add_vertex(frm, vlb1)
            if vlb2 != VACANT_VERTEX_LABEL:
                g.add_vertex(to, vlb2)
            g.add_edge(AUTO_EDGE_ID, frm, to, elb)
        return g

    def from_graph(self, g):
        """Build DFScode from graph `g`."""
        raise NotImplementedError('Not inplemented yet.')

    def build_rmpath(self):
        """Build right most path."""
        '''最右拓展不应该是通过DFS实现，不应该通过栈来实现吗？'''
        self.rmpath = list()
        old_frm = None
        for i in range(len(self) - 1, -1, -1):
            dfsedge = self[i]
            frm, to = dfsedge.frm, dfsedge.to
            if frm < to and (old_frm is None or to == old_frm):

                self.rmpath.append(i)
                old_frm = frm
        return self

    def get_num_vertices(self):
        """Return number of vertices in the corresponding graph."""
        return len(set(
            [dfsedge.frm for dfsedge in self] +
            [dfsedge.to for dfsedge in self]
        ))


class PDFS(object):
    """PDFS class."""
    '''
    PDFS 是 gSpan 算法中的一个辅助类，用来表示一个投影 DFS序列（Projected Depth First Search sequence）
    的节点信息。
    在 gSpan 算法中，每个频繁子图在原始图中的实例（投影）可以通过一系列的 PDFS 对象串联起来。
    '''
    '''
    gid：当前投影所在的原始图的 ID。
    edge：当前投影中涉及的边（对应 gSpan 中的 Edge 对象）。
    prev：指向该投影的前一个 PDFS 节点，用于串联成一个链表。
    
    嵌入图不应该是一个DFScode，怎么会只是一个edge呢？
    '''
    def __init__(self, gid=VACANT_GRAPH_ID, edge=None, prev=None):
        """Initialize PDFS instance."""
        self.gid = gid
        self.edge = edge
        self.prev = prev

'''将PDFS串起来的链表'''
class Projected(list):
    """Projected is a list of PDFS.

    Each element of Projected is a projection one frequent graph in one
    original graph.
    """

    def __init__(self):
        """Initialize Projected instance."""
        super(Projected, self).__init__()

    def push_back(self, gid, edge, prev):
        """Update this Projected instance."""
        self.append(PDFS(gid, edge, prev))
        return self
'''投影这一点和我想的一样'''

class History(object):
    """History class."""

    def __init__(self, g, pdfs):
        """Initialize History instance."""
        super(History, self).__init__()
        self.edges = list()
        self.vertices_used = collections.defaultdict(int)
        self.edges_used = collections.defaultdict(int)
        '''
        defaultdict 是 Python 的 collections 模块中的一种字典类型。与普通字典不同，defaultdict 提供了一个默认值生成器（default_factory），当访问一个不存在的键时，自动为该键生成一个默认值，而不会抛出 KeyError。
        2. collections.defaultdict(int)
        这里的 defaultdict(int) 使用 int 作为默认值生成器。
        当访问一个不存在的键时，defaultdict 会自动将该键的值初始化为 0（int() 的默认返回值）。
        '''
        if pdfs is None:
            return
        while pdfs:
            e = pdfs.edge
            self.edges.append(e) #访问过的边
            (self.vertices_used[e.frm],
                self.vertices_used[e.to],
                self.edges_used[e.eid]) = 1, 1, 1

            pdfs = pdfs.prev
        self.edges = self.edges[::-1]
        '''
        构造 self.edges 的过程
            pdfs 是一个链表结构（通过 prev 属性链接到前一个 PDFS 节点）。
            遍历 pdfs 时，从当前的 pdfs 节点一直回溯到第一个节点，将每条边 e 添加到 self.edges。
            因为是从末尾向起点回溯，所以 self.edges 的初始顺序是逆序的（最后的边在最前，第一条边在最后）。
        反转列表
            self.edges[::-1] 是 Python 列表的一种切片操作，用来生成一个反向排列的新列表。
            通过 self.edges = self.edges[::-1]，将 self.edges 转换为从起点到终点的顺序。
        '''

    def has_vertex(self, vid):
        """Check if the vertex with vid exists in the history."""
        return self.vertices_used[vid] == 1

    def has_edge(self, eid):
        """Check if the edge with eid exists in the history."""
        return self.edges_used[eid] == 1


class gSpan(object):
    """`gSpan` algorithm."""

    def __init__(self,
                 database_file_name,
                 min_support=10,
                 min_num_vertices=1,
                 max_num_vertices=float('inf'),
                 max_ngraphs=float('inf'),
                 is_undirected=True,
                 verbose=False,
                 visualize=False,
                 where=False):
        """Initialize gSpan instance."""
        self._database_file_name = database_file_name
        self.graphs = dict()
        self._max_ngraphs = max_ngraphs # 限制处理的最大图数量。当图数据库非常大时，可以通过设置此参数来控制算法的处理规模。
        self._is_undirected = is_undirected
        self._min_support = min_support
        self._min_num_vertices = min_num_vertices # 频繁子图中最少包含的顶点数量。用于过滤过小的子图。
        self._max_num_vertices = max_num_vertices
        self._DFScode = DFScode()
        self._support = 0
        self._frequent_size1_subgraphs = list()
        # Include subgraphs with
        # any num(but >= 2, <= max_num_vertices) of vertices.
        self._frequent_subgraphs = list()
        self._counter = itertools.count() # 为频繁子图分配唯一的 ID
        self._verbose = verbose # 保存详细输出和可视化的设置。
        self._visualize = visualize # 保存详细输出和可视化的设置。
        self._where = where
        self.timestamps = dict() # 时间戳
        if self._max_num_vertices < self._min_num_vertices:
            print('Max number of vertices can not be smaller than '
                  'min number of that.\n'
                  'Set max_num_vertices = min_num_vertices.')
            self._max_num_vertices = self._min_num_vertices
        self._report_df = pd.DataFrame() # 存储和报告算法运行结果，如频繁子图的详细信息、支持度等

    def time_stats(self):
        """Print stats of time."""
        func_names = ['_read_graphs', 'run']
        time_deltas = collections.defaultdict(float)
        for fn in func_names:
            time_deltas[fn] = round(
                self.timestamps[fn + '_out'] - self.timestamps[fn + '_in'],
                2
            )

        print('Read:\t{} s'.format(time_deltas['_read_graphs']))
        print('Mine:\t{} s'.format(
            time_deltas['run'] - time_deltas['_read_graphs']))
        print('Total:\t{} s'.format(time_deltas['run']))

        return self

    @record_timestamp
    def _read_graphs(self):
        self.graphs = dict()
        with codecs.open(self._database_file_name, 'r', 'utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
            tgraph, graph_cnt = None, 0
            for i, line in enumerate(lines):
                cols = line.split(' ')
                if cols[0] == 't':
                    if tgraph is not None:
                        self.graphs[graph_cnt] = tgraph
                        graph_cnt += 1
                        tgraph = None
                    if cols[-1] == '-1' or graph_cnt >= self._max_ngraphs:
                        break
                    tgraph = Graph(graph_cnt,
                                   is_undirected=self._is_undirected,
                                   eid_auto_increment=True)
                elif cols[0] == 'v':
                    tgraph.add_vertex(cols[1], cols[2]) # id and label
                elif cols[0] == 'e':
                    edgeLable=(cols[3],cols[4],cols[5])
                    tgraph.add_edge(AUTO_EDGE_ID, cols[1], cols[2], edgeLable)
                    # print(edgeLable)
            # adapt to input files that do not end with 't # -1'
            if tgraph is not None:
                self.graphs[graph_cnt] = tgraph
        return self

    @record_timestamp
    def _generate_1edge_frequent_subgraphs(self): #返回一个1顶点图
        vlb_counter = collections.Counter() # 统计每个定点标签的支持度
        vevlb_counter = collections.Counter()
        vlb_counted = set() # 被计数过的顶点标签
        vevlb_counted = set()
        # 对每个图的每个顶点，每条边，统计支持度
        for g in self.graphs.values():
            for v in g.vertices.values():
                if (g.gid, v.vlb) not in vlb_counted:
                    vlb_counter[v.vlb] += 1
                vlb_counted.add((g.gid, v.vlb))
                # 为什么不直接调DFSedge的vevlb呢？
                for to, e in v.edges.items():
                    vlb1, vlb2 = v.vlb, g.vertices[to].vlb
                    if self._is_undirected and vlb1 > vlb2:
                        vlb1, vlb2 = vlb2, vlb1
                    if (g.gid, (vlb1, e.elb, vlb2)) not in vevlb_counter:
                        vevlb_counter[(vlb1, e.elb, vlb2)] += 1
                    vevlb_counted.add((g.gid, (vlb1, e.elb, vlb2)))

        # add frequent vertices.
        for vlb, cnt in vlb_counter.items():
            if cnt >= self._min_support:
                g = Graph(gid=next(self._counter),
                          is_undirected=self._is_undirected)
                g.add_vertex(0, vlb)
                self._frequent_size1_subgraphs.append(g)
                if self._min_num_vertices <= 1:
                    self._report_size1(g, support=cnt)
            else:
                continue
        if self._min_num_vertices > 1:
            self._counter = itertools.count()

    @record_timestamp
    def run(self):
        """Run the gSpan algorithm."""
        '''
        先创建root字典，对每一个图，对图里的每一个顶点，调用前向拓展得到edges，对每一个edge，用root的键“eve”存储该eve的PDFS
        '''
        self._read_graphs()
        self._generate_1edge_frequent_subgraphs() # 找到频繁1顶点图，此函数不生效
        if self._max_num_vertices < 2:
            return
        root = collections.defaultdict(Projected) # root 的键是 (v_label, edge_label, to_label) 三元组。root 的值是 Projected，包含所有投影。
        for gid, g in self.graphs.items():
            for vid, v in g.vertices.items():
                edges = self._get_forward_root_edges(g, vid)
                for e in edges:
                    root[(v.vlb, e.elb, g.vertices[e.to].vlb)].append(
                        PDFS(gid, e, None)
                    ) # root[(v-e-v)]是一个list，键是所有的边，值是这些边出现的子图的id及边信息，同一个图一条eve只出现一次

        for vevlb, projected in root.items():
            self._DFScode.append(DFSedge(0, 1, vevlb))
            self._subgraph_mining(projected) # 子图投影，一个DFScode对应一个project
            self._DFScode.pop()

    def _get_support(self, projected):
        return len(set([pdfs.gid for pdfs in projected]))

    def _report_size1(self, g, support):
        g.display()
        print('\nSupport: {}'.format(support))
        print('\n-----------------\n')

    def _report(self, projected):
        self._frequent_subgraphs.append(copy.copy(self._DFScode))
        if self._DFScode.get_num_vertices() < self._min_num_vertices:
            return
        g = self._DFScode.to_graph(gid=next(self._counter),
                                   is_undirected=self._is_undirected)
        display_str = g.display()
        print('\nSupport: {}'.format(self._support))

        # Add some report info to pandas dataframe "self._report_df".
        self._report_df = self._report_df.append(
            pd.DataFrame(
                {
                    'support': [self._support],
                    'description': [display_str],
                    'num_vert': self._DFScode.get_num_vertices()
                },
                index=[int(repr(self._counter)[6:-1])]
            )
        )
        if self._visualize:
            g.plot()
        if self._where:
            print('where: {}'.format(list(set([p.gid for p in projected]))))
        print('\n-----------------\n')

    def _get_forward_root_edges(self, g, frm):
        result = []
        v_frm = g.vertices[frm]
        for to, e in v_frm.edges.items():
            if (not self._is_undirected) or v_frm.vlb <= g.vertices[to].vlb:
                result.append(e)
        return result

    def _get_backward_edge(self, g, e1, e2, history):
        '''
        当前已有一个正在扩展的子图。
        需要尝试找到一条后向边，将子图的 右侧路径 的某个顶点连接回 左侧路径 的顶点。

        该方法用于在图中查找 符合条件的后向边（Backward Edge）
        查找连接子图中某条边的目标顶点（e2.to）到源顶点（e1.frm）的后向边。
        在扩展子图时，后向边用于检查子图是否是当前最小 DFS 代码（is_min() 方法）。

        这个函数用于找到一条反向边，使得整体图的DFS代码最小
        g: 当前图对象。
        e1：DFS 代码中当前子图的一条边。
e2：DFS 代码中子图的另一条边，用于寻找连接的目标。
history：记录当前子图的顶点和边访问情况。
        history: 一个 History 对象，记录了当前子图中已经访问过的顶点和边。
        '''
        if self._is_undirected and e1 == e2:
            return None
        for to, e in g.vertices[e2.to].edges.items():
            '''
            该方法用于寻找符合条件的后向边，从而扩展当前子图，并确保生成的 DFS 代码（
            深度优先搜索代码）保持最小。
            '''
            if history.has_edge(e.eid) or e.to != e1.frm:
                continue
            # if reture here, then self._DFScodep[0] != dfs_code_min[0]
            # should be checked in _is_min(). or:
            if self._is_undirected:
                if e1.elb < e.elb or (
                        e1.elb == e.elb and
                        g.vertices[e1.to].vlb <= g.vertices[e2.to].vlb):
                    return e
            else:
                if g.vertices[e1.frm].vlb < g.vertices[e2.to].vlb or (
                        g.vertices[e1.frm].vlb == g.vertices[e2.to].vlb and
                        e1.elb <= e.elb):
                    return e
            # if e1.elb < e.elb or (e1.elb == e.elb and
            #     g.vertices[e1.to].vlb <= g.vertices[e2.to].vlb):
            #     return e
        return None

    def _get_forward_pure_edges(self, g, rm_edge, min_vlb, history):
        result = []
        for to, e in g.vertices[rm_edge.to].edges.items():
            if min_vlb <= g.vertices[e.to].vlb and ( # 目标顶点的标签（vlb）满足 min_vlb <= g.vertices[e.to].vlb。
                    not history.has_vertex(e.to)):
                result.append(e)
        return result

    def _get_forward_rmpath_edges(self, g, rm_edge, min_vlb, history):
        result = []
        to_vlb = g.vertices[rm_edge.to].vlb
        for to, e in g.vertices[rm_edge.frm].edges.items():
            new_to_vlb = g.vertices[to].vlb
            if (rm_edge.to == e.to or
                    min_vlb > new_to_vlb or
                    history.has_vertex(e.to)):
                continue
            if rm_edge.elb < e.elb or (rm_edge.elb == e.elb and
                                       to_vlb <= new_to_vlb):
                result.append(e)
        return result

    def _is_min(self):
        '''
        确定当前子图的 DFS 代码（_DFScode）是否是最小 DFS 代码。如果不是，则终止进一步挖掘该子图。
        '''
        if self._verbose:
            print('is_min: checking {}'.format(self._DFScode))
        if len(self._DFScode) == 1:
            return True
        g = self._DFScode.to_graph(gid=VACANT_GRAPH_ID,
                                   is_undirected=self._is_undirected)
        dfs_code_min = DFScode() # 一个列表
        root = collections.defaultdict(Projected)
        for vid, v in g.vertices.items():
            edges = self._get_forward_root_edges(g, vid)
            for e in edges:
                root[(v.vlb, e.elb, g.vertices[e.to].vlb)].append(
                    PDFS(g.gid, e, None))
        min_vevlb = min(root.keys())
        dfs_code_min.append(DFSedge(0, 1, min_vevlb))
        # No need to check if is min code because of pruning in get_*_edge*.

        def project_is_min(projected):
            dfs_code_min.build_rmpath()
            rmpath = dfs_code_min.rmpath
            min_vlb = dfs_code_min[0].vevlb[0]
            maxtoc = dfs_code_min[rmpath[0]].to

            backward_root = collections.defaultdict(Projected)
            flag, newto = False, 0,
            end = 0 if self._is_undirected else -1
            for i in range(len(rmpath) - 1, end, -1):
                if flag:
                    break
                for p in projected:
                    history = History(g, p)
                    e = self._get_backward_edge(g,
                                                history.edges[rmpath[i]],
                                                history.edges[rmpath[0]],
                                                history)
                    if e is not None:
                        backward_root[e.elb].append(PDFS(g.gid, e, p))
                        newto = dfs_code_min[rmpath[i]].frm
                        flag = True
            if flag:
                backward_min_elb = min(backward_root.keys())
                dfs_code_min.append(DFSedge(
                    maxtoc, newto,
                    (VACANT_VERTEX_LABEL,
                     backward_min_elb,
                     VACANT_VERTEX_LABEL)
                ))
                idx = len(dfs_code_min) - 1
                if self._DFScode[idx] != dfs_code_min[idx]:
                    return False
                return project_is_min(backward_root[backward_min_elb])

            forward_root = collections.defaultdict(Projected)
            flag, newfrm = False, 0
            for p in projected:
                history = History(g, p)
                edges = self._get_forward_pure_edges(g,
                                                     history.edges[rmpath[0]],
                                                     min_vlb,
                                                     history)
                if len(edges) > 0:
                    flag = True
                    newfrm = maxtoc
                    for e in edges:
                        forward_root[
                            (e.elb, g.vertices[e.to].vlb)
                        ].append(PDFS(g.gid, e, p))
            for rmpath_i in rmpath:
                if flag:
                    break
                for p in projected:
                    history = History(g, p)
                    edges = self._get_forward_rmpath_edges(g,
                                                           history.edges[
                                                               rmpath_i],
                                                           min_vlb,
                                                           history)
                    if len(edges) > 0:
                        flag = True
                        newfrm = dfs_code_min[rmpath_i].frm
                        for e in edges:
                            forward_root[
                                (e.elb, g.vertices[e.to].vlb)
                            ].append(PDFS(g.gid, e, p))

            if not flag:
                return True

            forward_min_evlb = min(forward_root.keys())
            dfs_code_min.append(DFSedge(
                newfrm, maxtoc + 1,
                (VACANT_VERTEX_LABEL, forward_min_evlb[0], forward_min_evlb[1]))
            )
            idx = len(dfs_code_min) - 1
            if self._DFScode[idx] != dfs_code_min[idx]:
                return False
            return project_is_min(forward_root[forward_min_evlb])

        res = project_is_min(root[min_vevlb])
        return res

    def _subgraph_mining(self, projected):
        self._support = self._get_support(projected)
        if self._support < self._min_support:
            return
        if not self._is_min():
            return
        self._report(projected)

        num_vertices = self._DFScode.get_num_vertices()
        self._DFScode.build_rmpath()
        rmpath = self._DFScode.rmpath
        maxtoc = self._DFScode[rmpath[0]].to
        min_vlb = self._DFScode[0].vevlb[0]

        forward_root = collections.defaultdict(Projected)
        backward_root = collections.defaultdict(Projected)
        for p in projected:
            g = self.graphs[p.gid]
            history = History(g, p)
            # backward
            for rmpath_i in rmpath[::-1]:
                e = self._get_backward_edge(g,
                                            history.edges[rmpath_i],
                                            history.edges[rmpath[0]],
                                            history)
                if e is not None:
                    backward_root[
                        (self._DFScode[rmpath_i].frm, e.elb)
                    ].append(PDFS(g.gid, e, p))
            # pure forward
            if num_vertices >= self._max_num_vertices:
                continue
            edges = self._get_forward_pure_edges(g,
                                                 history.edges[rmpath[0]],
                                                 min_vlb,
                                                 history)
            for e in edges:
                forward_root[
                    (maxtoc, e.elb, g.vertices[e.to].vlb)
                ].append(PDFS(g.gid, e, p))
            # rmpath forward
            for rmpath_i in rmpath:
                edges = self._get_forward_rmpath_edges(g,
                                                       history.edges[rmpath_i],
                                                       min_vlb,
                                                       history)
                for e in edges:
                    forward_root[
                        (self._DFScode[rmpath_i].frm,
                         e.elb, g.vertices[e.to].vlb)
                    ].append(PDFS(g.gid, e, p))

        # backward
        for to, elb in backward_root:
            self._DFScode.append(DFSedge(
                maxtoc, to,
                (VACANT_VERTEX_LABEL, elb, VACANT_VERTEX_LABEL))
            )
            self._subgraph_mining(backward_root[(to, elb)])
            self._DFScode.pop()
        # forward
        # No need to check if num_vertices >= self._max_num_vertices.
        # Because forward_root has no element.
        for frm, elb, vlb2 in forward_root:
            self._DFScode.append(DFSedge(
                frm, maxtoc + 1,
                (VACANT_VERTEX_LABEL, elb, vlb2))
            )
            self._subgraph_mining(forward_root[(frm, elb, vlb2)])
            self._DFScode.pop()

        return self
