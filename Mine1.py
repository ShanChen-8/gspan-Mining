'''
图数据结构：
	边集：(字典){id:vertex}
	顶点集：邻接表(字典){v_Id:e_list}
遍历边集，建立projected
辅助数据结构：
'''
'''
差：
    eve_lab的构造  
    DfsCode的构造，从PDfsCode
'''
from collections import defaultdict
def ReadGraph():
    return
class Edge():
	def __init__(self,eId,eLb,frmId,frmLb,toLb,toId,):
		self.eId=eId
		self.eLb=eLb
		self.frmId=frmId
		self.frmLb=frmLb
		self.toId=toId
		self.toLb=toLb
class Vertex():
	def __init__(self,vId,vLb):
		self.vId=vId
		self.vLb=vLb
		self.edges=dict()# 存储了从当前顶点（Vertex 实例）出发的所有边

class Graph():
	def __init__(self):
		self.vertices=dict()
		self.edges=dict()

	def addEdge(self,edge):
		self.edges[edge.eId]=edge
		self.vertices[edge.frmId]=edge
	def addVert(self,vertex):
		self.vertices[vertex.vId]=None
class PDfsEdge():
	def __init__(self, frmId,toId,eId,eveLab):
		self.frmId=frmId
		self.toId=toId
		self.eId=eId
		self.eveLab=eveLab

class PDfsCode(list):# 存储PDfsEdge
	def __init__(self):
        super().__init__()
	def ToGraph(self):
        # 从DfsCode读图

    def ToDfsCode():# 将PDfsCode的DfsCode信息提取出来
        self.DfsCode=[]
        for k in PDfsCode:
            self.DfsCode.append(k.eveLab)
        return self.DfsCode

class PDfsCoLi(list):# 存储PDfsCode的列表，便于for操作，建立高阶project
	def __init__(self):# 例如PDfsCoLi.append(PDFSCode)

class Projected(defaultdict): #{“DfsCode”:PDfsCoLi}
	def __init__(self):
        super().__init__()
    # 对于一个拓展到的PDfsCode，检查DfsCode是否存在于Projected中，存在就append，不存在就新建
    def add_PdfsCode(PDfsCode,DfsCode):
        if DfsCode in self:
            self[DfsCode].append(PDfsCode)
        else:
            self[DfsCode]=PDfsCode

class SubGraphs(defaultdict): # 子图的DfsCode及它们的支持度
	def __init__(self):
	def add(self,DfsCode,support):
		self[DfsCode]=support
class Visited(set):
	def __init__(self):
		self.vertices={}
		self.edges={}
	def GraphIn(self,g):# 把graph写入Visited
	def VertexIn(self): # 把vertex写入Visited
	def EdgeIn(self): # 把edge写入Visited
class queue(list):
	def __init__(self):
	def append(self,a):
		queue.append(a)
	def deq(self):
		a=queue.pop(0)
		return a
# 辅助算法：
# 对一个图找到Min_DfsCode
def find_min(G,queue)
	# 如果对每个元素都无法最右拓展，返回最小DfsCode
	flag=0
	for i in range(len(queue)):
		if rmpath(queue[i],G,visited=Visited())
			flag=1
			break
	if flag==0:# 可拓展
		return min(queue)
	else: #至少一个可拓展
		for i in range(len(queue)):
			top=queue.deq() # 先出队，拓展之后再入队
			visited=Visited()
			while rmpath(G.top,visited):
				e=rmpath(G.top,visited) # 给出一条拓展边
				queue.append(e)
			find_min(G,queue)
# 先反向拓展，如若无，再前向拓展
def rmpath(PDfsCode,G,visited):
	g=PDfsCode.toGraph()
	visited.GraphIn(g)
	for ve in G.vertises[PDfsCode[-1].to]
		v=ve.vertex
		e=ve.edge
		if vertex in visited.vertises and e not in visited.edges
			visited.EdgeIn(e)
			return e
	# 没有反向边，寻找前向边
	for ve in G.vertices[DfsCode[-1].to]:
		v=ve.vertex
		e=ve.edge
		if v not in visited.vertices
			visited.VertexIn(v)
			visited.EdgeIn(v)
			return e
	return None
# 对projected挖掘，输入是projected，针对指针递归，在递归过程中修改频繁集
# 出口：projected == Null
def SubMining(projected,G)
	if projected == Null
		return  # 原地修改
	for DfsCode ,PDfsCoLi in projected.items():
		#  构建新的projected
		projected_newjected()
		fot PDfsCoDe in PDfsCiLi:
			visited=Visited()
			g=PDfsCode.ToGraph
			visited.GraphIn(g)
			while  rmpath(g.PdfsCode[-1],visited)
				e=rmpath(g.PDfsCode[-1],visited)
				PDfsCode.append(e)
				projected_new[DfsCode].append(PDfsCode)
		# 在这个DfsCode进行最右拓展已经完成了，现在筛选出大于频繁度的最小序列
		projected_new2=projected()
		for key,val in projected_new.items():
			g=val[0].ToGraph
			dfs_min=find_min(g)
			if dfs_min==key and len(val)>=support
				projected_new2[key]=val;
				MinedGraphs[DfsCode]=len(val)
		# 在最小频繁集上继续挖掘
		SubMining(projected_new2,G)


主函数
MinedSubGraphs=SubGraphs()
Projected1=Projected()
遍历边集，建立一阶边的Projected1，键DfsCode只使用eve_lab。
# 接下来对Projected1的每个PDfs挖掘
# 对图处理得到一阶图，构造递归的初始条件
for DfsCode,PDfsCoLi in Projected.items():
	support= len(PDfsCoLi)
	if support>supp_thres
		MinedSubGraphs.add(DfsCode,support)
		Projected2= Projected()
		for PDfsCode in PDfsCoLi:
			visited=Visited()
			找到在这个PDfsCode上的所有rmpath，append到Projected2对应的DfsCode
SubMining(projected,G)
