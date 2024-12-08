from collections import defaultdict
import copy

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
        queue=Queue()
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
            eveLb=(edge.frmLb,edge.eLb,edge.toLb)
            dfs_code.append(eveLb)
        return tuple(dfs_code)
    def intigralKey(self):
        temp=[]
        for edge in self:
            temp.append((edge.frmId,edge.toId,edge.eId,(edge.frmLb,edge.eLb,edge.toLb)))
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
        self.vertices = set() # id
        self.edges = set() # id

    def GraphIn(self, g):  # 把graph写入Visited
        for eId, edge in g.edges.items():
            self.edges.add(eId)
            self.vertices.add(edge.frmId)
            self.vertices.add(edge.toId)

    def EdgeIn(self, e):  # 把edge写入Visited
        self.edges.add(e.eId)
        self.vertices.add(e.frmId)
        self.vertices.add(e.toId)

class Queue(list): #存储PDfsCode
    def __init__(self):
        super().__init__()

    def inq(self, a):
        self.append(a)

    def deq(self):
        return self.pop(0)

    def ToDfsQueue(self):
        tem_queue=[]
        for p_dfs_code in self:
            a=p_dfs_code.ToDfsCode()
            tem_queue.append(a)
        return tem_queue



# 反向拓展，如若无，再前向拓展
def rmpath(G, p_dfs_code, visited):
    g = p_dfs_code.ToGraph()
    visited.GraphIn(g)
    for eId,edge in G.vertices[p_dfs_code[-1].toId].edges.items():
        if edge.toId in visited.vertices and eId not in visited.edges:
            visited.EdgeIn(edge)
            print('寻找到反向边{}，从{}到{}'.format(edge.eId,edge.frmId,edge.toId))
            return edge
    # 没有反向边，寻找前向边
    for eId,edge in G.vertices[p_dfs_code[-1].toId].edges.items():
        if edge.toId not in visited.vertices:
            # visited.VertexIn(G.vertices[edge.toId])
            visited.EdgeIn(edge)
            print('寻找到前向边{}，从{}到{}'.format(edge.eId,edge.frmId,edge.toId))
            return edge
    print("没法继续最右拓展")  # 确认返回 None
    return None

# 辅助算法：对一个图找到Min_DfsCode,首先需要将每条边放到queue中
def findMin(g, queue):
    # 如果对每个元素都无法最右拓展，返回最小DfsCode
    h_=len(queue)
    for i in range(h_):
        h=i
        queue_i=copy.copy(queue[i])
        e=rmpath( g,queue[i], visited=Visited())
        if e is not None:
            break # 说明可拓展
    if e is None:  # 不可拓展
        tem_queue=queue.ToDfsQueue()
        print('当前子图的最小路径为{}'.format(min(tem_queue)))
        return min(tem_queue)

    else:  # 至少一个可拓展
        for i in range(h+1):
            queue.deq()  # 有下一层，出队，上一层的节点全出去
        queue_i.append(e)
        queue.inq(queue_i)
        visited = Visited()
        for i in range(h_-h-1):
            top=queue.deq()
            e = rmpath(g, top, visited)# 给出一条拓展边
            if e is not None:
                top.append(e)
                queue.inq(top)
        return findMin(g, queue)

# 对projected挖掘，输入是projected，针对指针递归，在递归过程中修改频繁集
def SubMining(minedSubGraphs,projected,G,supp_thres):
    if not projected:
        return  print('挖掘完毕')# 原地修改
    nu=1
    for DfsCode, PDfsCoLi in projected.items():
        if len(DfsCode)==1:
            all_iter=len(projected)
            print('进度为{}/{}'.format(nu,all_iter))
            nu+=1
        #  构建新的projected
        projected_new = Projected()
        for p_dfs_code in PDfsCoLi:
            print('迭代对象为i.eId,i.frmId,i.toId {}'.format([(i.eId,i.frmId,i.toId) for i in p_dfs_code]))
            visited = Visited()
            g = p_dfs_code.ToGraph()
            visited.GraphIn(g)
            while True:
                e = rmpath(G,p_dfs_code, visited)
                if e is None:
                    break
                #     如果返回结果通过引用传递，先缓存，再判断是否break
                print("rmpath returned:e.eId,e.frmId,e.toId:", e.eId,e.frmId,e.toId)  # 检查返回值
                PCoCp=copy.copy(p_dfs_code)
                PCoCp.append(e)
                DfsCoCp=PCoCp.ToDfsCode()
                print('拓展以后的迭代对象为i.eId,i.frmId,i.toId{}'.format([(i.eId,i.frmId,i.toId) for i in PCoCp]))
                print('拓展以后的迭代对象为{}'.format(DfsCoCp))
                projected_new.add_PdfsCode(PCoCp, DfsCoCp)

        # 在这个DfsCode进行最右拓展已经完成了，现在筛选出大于频繁度的最小序列
        print('开始筛选子图')
        projected_filt = Projected()
        for key, val in projected_new.items():
            g = val[0].ToGraph()
            queue=g.inQueue()
            print('队列中有：{}'.format([i.ToDfsCode() for i in queue]))
            dfs_min = findMin(g, queue)
            if dfs_min == key and len(val) >= supp_thres:
                print('当前路径是最短路径')
                projected_filt[key] = val
                inti_key=val[0].intigralKey()
                minedSubGraphs[inti_key] = len(val)
                print('挖掘到子图{}\n支持度{}'.format(inti_key,len(val)))
            else:
                print('当前路径不是最短路径')

        # 在最小频繁集上继续挖掘
        SubMining(minedSubGraphs,projected_filt,G,supp_thres)
def ReadGraph(filepath):
    G = Graph()  # 初始化一个Graph对象
    with open(filepath, 'r') as file:
        lines = file.readlines()
        current_graph = G
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':  # 读取顶点信息
                vId = int(parts[1])
                vLb = parts[2]  # 顶点标签
                vertex = Vertex(vId, vLb)
                current_graph.addVert(vertex)
            elif parts[0] == 'e':  # 读取边信息
                frmId = int(parts[1])
                toId = int(parts[2])
                eId = len(current_graph.edges) + 1  # 自动生成边ID
                eLb = "-".join(parts[3:])  # 边的标签
                edge = Edge(eId, eLb, frmId, current_graph.vertices[frmId].vLb,
                            current_graph.vertices[toId].vLb, toId)
                current_graph.addEdge(edge)
    return G

# 将minedSubGraphs中的数据写入字典并保存为JSON文件
import json
def save_mined_graphs(mined_graphs, output_path):
    mined_graphs_dict = {}
    for key, support in mined_graphs.items():
        mined_graphs_dict[str(key)] = support  # 将key转换为字符串，以便可以序列化为JSON

    with open(output_path, 'w') as json_file:
        json.dump(mined_graphs_dict, json_file, indent=4)
    print(f"Mined subgraphs have been saved to {output_path}")
# 主函数
minedSubGraphs = SubGraphs()
projected = Projected()
G = ReadGraph("D:\软件\python及系列软件\环境测试\gSpan-master3\merged_graph2.data")  # 读取图
supp_thres=1
#对G中的每一条边,放入projected,键DfsCode只使用eve_lab。然后统计筛选
for eId,edge in G.edges.items():
    p_dfs_code=PDfsCode([edge])
    a=p_dfs_code.ToDfsCode()
    projected.add_PdfsCode(p_dfs_code,a)
# 统计筛选
projected_filt=Projected()
for key ,val in projected.items():
    if len(val) >=supp_thres:# 挖掘到的子图
        projected_filt[key] = val
        inti_key=val[0].intigralKey()
        minedSubGraphs[inti_key] = len(val)
all_iter=len(minedSubGraphs)
SubMining(minedSubGraphs,projected,G,supp_thres)
output_file_path = "mined_subgraphs.json"
save_mined_graphs(minedSubGraphs, output_file_path)

