from collections import defaultdict
import copy
import json
import time

'''在数据集上跑'''
'''每个进程保存文件'''
'''优化G全局变量'''
'''注释掉了所有minedSubGraphs，节省内存'''
'''全局变量共享'''
# 全局变量，用于子进程访问
# G = None


def init_pool(shared_mem):
    global shared # manager.dict()
    shared=shared_mem
    print("init_pool has been called and shared: G is set.")

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
        self.edges = dict()  # 存储了从当前顶点（Vertex 实例）出发的所有边{eId:edge}


class Graph():
    def __init__(self):
        self.vertices = dict()  # 存储所有顶点
        self.edges = dict()  # 存储所有边

    def addEdge(self, edge):
        self.edges[edge.eId] = edge
        if edge.frmId not in self.vertices:
            self.vertices[edge.frmId] = Vertex(edge.frmId, edge.frmLb)
        self.vertices[edge.frmId].edges[edge.eId] = edge
        if edge.toId not in self.vertices:
            self.vertices[edge.toId] = Vertex(edge.toId, edge.toLb)

    def addVert(self, vertex):
        self.vertices[vertex.vId] = vertex

    def inQueue(self):
        queue = Queue()
        for edge in self.edges.values():
            queue.inq(PDfsCode([edge]))
        return queue


class PDfsCode(list):  # 定义：projected-DfsCod，存储Edge
    def __init__(self, edges=None):
        super().__init__()
        if edges is not None:
            self.extend(edges)

    def ToGraph(self):
        # 从DfsCode读图
        g = Graph()
        for edge in self:
            g.addEdge(edge)

        return g

    def ToDfsCode(self):  # 将PDfsCode的DfsCode信息提取出来
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


class PDfsCoLi(list):  # 存储PDfsCode的列表，便于for操作，建立高阶project
    def __init__(self, p_codes=None):
        super().__init__()
        if p_codes is not None:
            self.extend(p_codes)


class Projected(dict):  # {“DfsCode”: PDfsCoLi}
    def __init__(self):
        super().__init__()

    def add_PdfsCode(self, p_dfs_code, dfs_code):
        if dfs_code in self:
            self[dfs_code].append(p_dfs_code)
        else:
            self[dfs_code] = PDfsCoLi([p_dfs_code])


class SubGraphs(dict):  # 子图的DfsCode及它们的支持度
    def __init__(self):
        super().__init__()

    def add(self, dfs_code, support):
        self[dfs_code] = support


class Visited():
    def __init__(self):
        self.vertices = set()  # id
        self.edges = set()  # id

    def GraphIn(self, g):  # 把graph写入Visited
        for eId, edge in g.edges.items():
            self.edges.add(eId)
            self.vertices.add(edge.frmId)
            self.vertices.add(edge.toId)

    def EdgeIn(self, e):  # 把edge写入Visited
        self.edges.add(e.eId)
        self.vertices.add(e.frmId)
        self.vertices.add(e.toId)


class Queue(list):  # 存储PDfsCode
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


# 反向拓展，如若无，再前向拓展
def rmpath(G, p_dfs_code, visited):
    g = p_dfs_code.ToGraph()
    visited.GraphIn(g)
    for eId, edge in G.vertices[p_dfs_code[-1].toId].edges.items():
        if edge.toId in visited.vertices and eId not in visited.edges:
            visited.EdgeIn(edge)
            return edge
    # 没有反向边，寻找前向边
    for eId, edge in G.vertices[p_dfs_code[-1].toId].edges.items():
        if edge.toId not in visited.vertices:
            # visited.VertexIn(G.vertices[edge.toId])
            visited.EdgeIn(edge)
            # print('寻找到前向边{}，从{}到{}'.format(edge.eId,edge.frmId,edge.toId))
            return edge
    return None


# 辅助算法：对一个图找到Min_DfsCode,首先需要将每条边放到queue中
def findMin(g, queue):
    # 如果对每个元素都无法最右拓展，返回最小DfsCode
    h_ = len(queue)
    for i in range(h_):
        h = i
        queue_i = copy.copy(queue[i])
        e = rmpath(g, queue[i], visited=Visited())
        if e is not None:
            break  # 说明可拓展
    if e is None:  # 不可拓展
        tem_queue = queue.ToDfsQueue()
        # print('当前子图的最小路径为{}'.format(min(tem_queue)))
        return min(tem_queue)

    else:  # 至少一个可拓展
        for i in range(h + 1):
            queue.deq()  # 有下一层，出队，上一层的节点全出去
        queue_i.append(e)
        queue.inq(queue_i)
        visited = Visited()
        for i in range(h_ - h - 1):
            top = queue.deq()
            e = rmpath(g, top, visited)  # 给出一条拓展边
            if e is not None:
                top.append(e)
                queue.inq(top)
        return findMin(g, queue)


# 对projected挖掘，输入是projected，针对指针递归，在递归过程中修改频繁集
def SubMining(minedSubGraphs, projected, G, supp_thres, out_filename):
    if not projected:
        return  # 原地修改

    for DfsCode, PDfsCoLi in projected.items():
        # 这里应该检查是否是最小的，非最小没有必要拓展了，在之前的projected里已经确认是最小了
        projected_new = Projected()
        for p_dfs_code in PDfsCoLi:
            visited = Visited()
            g = p_dfs_code.ToGraph()  # 用来构造visited
            visited.GraphIn(g)
            while True:
                e = rmpath(G, p_dfs_code, visited)
                if e is None:
                    break
                #     如果返回结果通过引用传递，先缓存，再判断是否break
                PCoCp = copy.copy(p_dfs_code)
                PCoCp.append(e)
                DfsCoCp = PCoCp.ToDfsCode()
                projected_new.add_PdfsCode(PCoCp, DfsCoCp)

        projected_filt = Projected()
        t = 0
        for key, val in projected_new.items():
            t += 1
            g = val[0].ToGraph()
            queue = g.inQueue()
            dfs_min = findMin(g, queue)
            if dfs_min == key and len(val) >= supp_thres:
                # print('当前路径是最短路径')
                inti_key = val[0].intigralKey()
                if len(inti_key) < 11:  # 最多到10阶子图
                    projected_filt[key] = val
                    # minedSubGraphs[inti_key] = len(val)
                    save_mined_graph_incrementally(inti_key, len(val), out_filename)
                    print('挖掘到{}阶子图{}，，支持度{}'.format(len(inti_key), inti_key, len(val)))
                    print(f'此子图处于projected中的：{t}/{len(projected_new)}')

        # 在最小频繁集上继续挖掘
        SubMining(minedSubGraphs, projected_filt, G, supp_thres, out_filename)


def ReadGraph(filepath):
    t = 0
    G = Graph()  # 初始化一个Graph对象
    with open(filepath, 'r') as file:
        lines = file.readlines()
        current_graph = G
        for line in lines:
            t += 1
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':  # 读取顶点信息
                vId = int(parts[1])
                if parts[2] == 'Mike':
                    vLb = 0
                elif parts[2] == 'John':
                    vLb = 1
                else:
                    vLb = 2

                # vLb = parts[2]  # 顶点标签
                vertex = Vertex(vId, vLb)
                if t % 10000 == 0:
                    print('读取第{}行信息，为v，vId:{},vLb:{}'.format(t, vId, vLb))
                    print('读取进度为：{}/{}'.format(t, len(lines)))
                current_graph.addVert(vertex)
            elif parts[0] == 'e':  # 读取边信息
                if parts[4] == 'strategy_name-1':
                    strat = 1
                elif parts[4] == 'strategy_name-2':
                    strat = 2
                elif parts[4] == 'strategy_name-3':
                    strat = 3
                elif parts[4] == 'strategy_name-4':
                    strat = 4
                elif parts[4] == 'strategy_name-5':
                    strat = 5
                else:
                    strat = 6
                if parts[5] == 'buscode1':
                    bus = 1
                elif parts[5] == 'buscode2':
                    bus = 2
                else:
                    bus = 3
                amt = int(parts[3])
                frmId = int(parts[1])
                toId = int(parts[2])
                eId = len(current_graph.edges) + 1  # 自动生成边ID
                eLb = " ".join([str(amt), str(strat), str(bus)])  # 边的标签
                edge = Edge(eId, eLb, frmId, current_graph.vertices[frmId].vLb,
                            current_graph.vertices[toId].vLb, toId)
                if t % 10000 == 0:
                    print('读取第{}行信息，为e,eId:{},eLb:{}'.format(t, eId, eLb))
                    print('读取进度为：{}/{}'.format(t, len(lines)))
                current_graph.addEdge(edge)
    return G


# 将minedSubGraphs中的数据写入字典并保存为JSON文件
def save_mined_graphs(mined_graphs, output_path):
    mined_graphs_dict = {}
    for key, support in mined_graphs.items():
        mined_graphs_dict[str(key)] = support  # 将key转换为字符串，以便可以序列化为JSON

    with open(output_path, 'w') as json_file:
        json.dump(mined_graphs_dict, json_file, indent=4)
    print(f"Mined subgraphs have been saved to {output_path}")


import json
import os


def save_mined_graph_incrementally(key, support, output_path):
    """
    将挖掘的子图增量存入 JSON 文件。
    如果文件已存在，则加载现有内容并追加新数据。
    如果文件不存在，则创建新文件。
    """
    if os.path.exists(output_path):
        # 文件存在，加载现有内容
        with open(output_path, 'r') as json_file:
            mined_graphs_dict = json.load(json_file)
    else:
        # 文件不存在，初始化为空字典
        mined_graphs_dict = {}

    # 添加新挖掘的子图
    mined_graphs_dict[str(key)] = support

    # 写回文件
    with open(output_path, 'w') as json_file:
        json.dump(mined_graphs_dict, json_file, indent=4, ensure_ascii=False)


from multiprocessing import Pool, cpu_count


def SubMiningTask(args):
    """
    包装 SubMining 函数，用于多进程执行。
    """
    key, val, supp_thres, i, length, out_filename = args  # 解包参数
    G=shared["Graph"]
    print(f'正在处理任务 {i}/{length}，一阶子图 {key}')
    minedSubGraphs2 = SubGraphs()  # 初始化子图存储
    projected_filt_ = Projected()  # 初始化筛选后的投影
    projected_filt_[key] = val
    try:
        SubMining(minedSubGraphs2, projected_filt_, G, supp_thres, out_filename)  # 使用全局 G
        print(f'任务 {i}/{length} 完成')
        # return minedSubGraphs2
        return None
    except Exception as e:
        print(f'任务 {i}/{length} 处理时出错: {e}')
        # return minedSubGraphs2
        return None

from multiprocessing import Manager, Pool
def ParallelSubMining(minedSubGraphs, projected_filt, G_local, supp_thres):
    """
    使用多进程加速对 projected_filt 的处理。
    """
    # 准备参数列表，不传递 G_local
    args_list = [
        (key, val, supp_thres, i, len(projected_filt), f"./t2/mined_subgraphs1_{i}.json")
        for i, (key, val) in enumerate(projected_filt.items(), start=1)
    ]
    # 使用 Manager 创建共享对象
    with Manager() as manager:
        # 创建一个共享字典来存储复杂数据
        shared_memory = manager.dict()
        shared_memory["Graph"] = G_local  # 存储复杂数据结构
        # 创建进程池，并传递初始化函数和参数
        # with Pool(processes=int(cpu_count()*0.3), initializer=init_pool, initargs=(G_local,)) as pool:
        with Pool(processes=20, initializer=init_pool, initargs=(shared_memory,)) as pool:
            results = pool.map(SubMiningTask, args_list)

    # 合并所有子进程的结果到 minedSubGraphs
    # for mSG in results:
    #     minedSubGraphs.update(mSG)


# 主函数
if __name__ == '__main__':
    # 读取图和初始化
    minedSubGraphs = SubGraphs()
    projected = Projected()
    G = ReadGraph("/root/Mining/merged_graph.data")  # 读取图
    supp_thres = 10000

    # 对 G 中的每一条边, 放入 projected
    t = 0
    starttime = time.time()
    for eId, edge in G.edges.items():
        t += 1
        p_dfs_code = PDfsCode([edge])
        a = p_dfs_code.ToDfsCode()
        projected.add_PdfsCode(p_dfs_code, a)
        if t % 10000 == 0:
            print(f'处理到了第{t}/{len(G.edges)}条边')

    # 统计筛选
    projected_filt = Projected()
    for i, (key, val) in enumerate(projected.items()):
        if len(val) >= supp_thres:  # 挖掘到的子图
            projected_filt[key] = val
            inti_key = val[0].intigralKey()
            minedSubGraphs[inti_key] = len(val)

    # 保存初步结果
    for i, (key, val) in enumerate(projected_filt.items()):
        inti_key=val[0].intigralKey()
        save_mined_graph_incrementally(inti_key, len(val), f"./t2/mined_subgraphs1_{i + 1}.json")

    # 并行挖掘
    ParallelSubMining(minedSubGraphs, projected_filt, G, supp_thres)
    endtime = time.time()
    print(f'time:{starttime}-{endtime}')

    # 保存最终结果
    output_file_path = "./t2/mined_subgraphs4_4_onetime1.json"
    save_mined_graphs(minedSubGraphs, output_file_path)



