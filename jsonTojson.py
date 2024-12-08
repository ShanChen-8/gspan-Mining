import json
import ast

# 给定的子图字典
data = {
    "((474510, 790403, 161, (1, '0 2 1', 0)),)": 36822,
    "((474510, 790403, 161, (1, '0 2 1', 0)), (790403, 653016, 2837716, (0, '0 2 1', 0)))": 12001,
    "((474510, 790403, 161, (1, '0 2 1', 0)), (790403, 427936, 4914182, (0, '0 2 3', 1)))": 12302,
    "((618119, 375903, 287, (1, '0 2 1', 0)), (375903, 523573, 21607, (0, '0 2 2', 0)))": 11971,
    "((618119, 375903, 287, (1, '0 2 1', 0)), (375903, 505360, 50934, (0, '0 3 2', 0)))": 24218,
    "((618119, 375903, 287, (1, '0 2 1', 0)), (375903, 718353, 2449845, (0, '0 3 3', 0)))": 32418,
}

# 顶点标签映射
def get_vertex_name(label):
    if label == 0:
        return "Mike"
    elif label == 1:
        return "John"
    else:
        return "Jobs"

# 转换字典为所需的 JSON 格式
json_list = []

for key, frequency in data.items():
    # 将字符串转换为 Python 元组
    edges_tuple = ast.literal_eval(key)
    nodes_dict = {}  # 存储节点的唯一性 {vId: node_id}
    nodes = []
    edges = []

    node_id_counter = 0
    for edge in edges_tuple:
        frmId, toId, eId, labels = edge
        frmLabel, edgeLabel, toLabel = labels

        # 处理起点节点
        if frmId not in nodes_dict:
            nodes_dict[frmId] = str(node_id_counter)
            nodes.append({
                "node_id": str(node_id_counter),
                "name": get_vertex_name(frmLabel)
            })
            node_id_counter += 1

        # 处理终点节点
        if toId not in nodes_dict:
            nodes_dict[toId] = str(node_id_counter)
            nodes.append({
                "node_id": str(node_id_counter),
                "name": get_vertex_name(toLabel)
            })
            node_id_counter += 1

        # 处理边
        amt, strategy_name, buscode = edgeLabel.split()
        edges.append({
            "source_node_id": nodes_dict[frmId],
            "target_node_id": nodes_dict[toId],
            "amt": amt,
            "strategy_name": strategy_name,
            "buscode": buscode
        })

    # 构造最终 JSON 数据
    json_list.append({
        "frequency": frequency,
        "nodes": nodes,
        "edges": edges
    })

# 将结果转换为 JSON 格式并打印
json_result = json.dumps(json_list, indent=2)
print(json_result)
