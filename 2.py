import csv
import networkx as nx
import os
import json
import random
from tqdm import tqdm

def load_nodes(nodes_file):
    nodes = {}
    with open(nodes_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            node_id = int(row['key'])
            nodes[node_id] = {
                'type': row['type'],
                'code': row['code'],
                'location': row['location'],
                'functionId': row['functionId'],
                'isCFGNode': row['isCFGNode'] == 'True',
                'operator': row['operator'],
                'baseType': row['baseType'],
                'completeType': row['completeType'],
                'identifier': row['identifier'],
            }
    return nodes


def load_cfg_edges(edges_file):
    edges = []
    with open(edges_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row['type'] != 'FLOWS_TO':
                continue
            edges.append((int(row['start']), int(row['end'])))
    return edges


def build_cfg(nodes_file, edges_file):
    nodes = load_nodes(nodes_file)
    edges = load_cfg_edges(edges_file)

    G = nx.DiGraph()
    for nid, info in nodes.items():
        if info['isCFGNode'] or info['type'].startswith('CFG'):
            G.add_node(nid, **info)

    for s, e in edges:
        if s in G and e in G:
            G.add_edge(s, e)

    entry = next(n for n, d in G.nodes(data=True) if d['type'] == 'CFGEntryNode')
    exit_ = next(n for n, d in G.nodes(data=True) if d['type'] == 'CFGExitNode')
    return G, entry, exit_


def find_all_paths(G, start, end, max_paths=500, cutoff=80, max_steps=50000):
    """
    手写 DFS：
      - 最多 max_paths 条路径
      - 路径长度 <= cutoff
      - 总扩展步数 <= max_steps（防止卡死）
    """
    paths = []
    stack = [(start, [start])]
    steps = 0

    while stack:
        node, path = stack.pop()
        steps += 1
        if steps > max_steps:
            print(f"[WARN] path search aborted: exceed max_steps={max_steps}")
            break

        # 长度超了就不再往下扩展
        if len(path) > cutoff:
            continue

        if node == end:
            paths.append(path)
            if max_paths is not None and len(paths) >= max_paths:
                break
            # 到了 end，不再往后扩展
            continue

        for nbr in G.successors(node):
            # 保证简单路径：不走回头路
            if nbr in path:
                continue
            stack.append((nbr, path + [nbr]))

    return paths



def find_common_prefix(paths):
    if not paths:
        return []
    prefix = paths[0]
    for path in paths[1:]:
        new_prefix = []
        for i in range(min(len(prefix), len(path))):
            if prefix[i] == path[i]:
                new_prefix.append(prefix[i])
            else:
                break
        prefix = new_prefix
    return prefix


def remove_common_prefix(paths, common_prefix):
    unique_paths = []
    for path in paths:
        unique_paths.append(path[len(common_prefix):])
    return unique_paths


def merge_paths(paths):
    common_prefix = find_common_prefix(paths)
    unique_parts = remove_common_prefix(paths, common_prefix)
    return common_prefix, unique_parts


def load_label_map_from_magnet(magnet_json_path):
    """
    从 /root/AMB/MAGNET-main/one_step_run/dateset/Magnet_devign.json
    构造 file_name(如 71635c47f...e4e0.c) -> target(0/1) 的字典
    """
    with open(magnet_json_path, "r") as f:
        data = json.load(f)

    label_map = {}
    for item in data:
        raw_path = item["file_path"]          # 例如 "devign_raw_code2/71635c47f...e4e0.c"
        fname = os.path.basename(raw_path)    # 取出 "71635c47f...e4e0.c"
        label_map[fname] = int(item["target"])
    return label_map


def save_paths_meta_compressed(nodes_file, edges_file, paths, output_root, real_label):
    nodes = load_nodes(nodes_file)
    src_code_file = None

    for info in nodes.values():
        if info['type'] == 'File':
            src_code_file = info['code']
            break

    unique_paths = []
    seen = set()
    for p in paths:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            unique_paths.append(p)

    common_prefix, unique_suffixes = merge_paths(unique_paths)

    cpg_dir = os.path.basename(os.path.dirname(nodes_file))

    meta = {
        "cpg_dir": cpg_dir,
        "src_code_file": src_code_file,
        "nodes_file": nodes_file,
        "edges_file": edges_file,
        "real_label": int(real_label),
        "common_prefix": common_prefix,
        "paths": []
    }

    for pid, suffix in enumerate(unique_suffixes):
        meta["paths"].append({
            "path_id": pid,
            "suffix": suffix,
            "soft_label": 0.0
        })

    os.makedirs(output_root, exist_ok=True)
    out_json = os.path.join(output_root, f"{cpg_dir}_paths.json")
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)


def process_sample_files(code_file_dir, parsed_file_dir, magnet_json, output_dir, num_samples=None):
    # 1) 先把 file_name -> label 映射读出来
    label_map = load_label_map_from_magnet(magnet_json)

    # 2) 获取源代码文件（排序，方便复现 & debug）
    all_code_files = sorted(f for f in os.listdir(code_file_dir) if f.endswith(".c"))
    if not all_code_files:
        print(f"[WARN] 源代码目录为空: {code_file_dir}")
        return

    # 如果指定了 num_samples，就截断；否则全量
    if num_samples is not None:
        all_code_files = all_code_files[:num_samples]

    total = len(all_code_files)
    print(f"[INFO] 准备处理 {total} 个 .c 文件")

    for idx, code_file in enumerate(tqdm(all_code_files, desc="Processing Samples", unit="file")):
        # 关键：每个文件先打印一行，卡住时能知道卡在哪个文件
        print(f"[INFO] ({idx+1}/{total}) start {code_file}", flush=True)

        # code_file 形如 "71635c47fc9446a6fbaf92a4bd44a4e0.c"
        if code_file not in label_map:
            print(f"[SKIP] {code_file} because label not found in Magnet_devign.json")
            continue
        real_label = label_map[code_file]

        # 注意：parsed 这边的目录名就是同名文件夹，不要再拼接 '.c'
        parsed_file_path = os.path.join(parsed_file_dir, code_file)
        if not os.path.isdir(parsed_file_path):
            print(f"[SKIP] {code_file} as parsed folder {parsed_file_path} doesn't exist.")
            continue

        nodes_file = os.path.join(parsed_file_path, 'nodes.csv')
        edges_file = os.path.join(parsed_file_path, 'edges.csv')

        if not (os.path.exists(nodes_file) and os.path.exists(edges_file)):
            print(f"[SKIP] {code_file} because nodes/edges.csv missing.")
            continue

        try:
            graph, entry, exit_ = build_cfg(nodes_file, edges_file)

            # 可选：图太大就直接跳过，避免 all_simple_paths 爆炸
            n_nodes = graph.number_of_nodes()
            n_edges = graph.number_of_edges()
            if n_nodes > 800 or n_edges > 2000:
                print(f"[SKIP-LARGE] {code_file}: graph too large (nodes={n_nodes}, edges={n_edges})")
                continue

            # 使用带 cutoff 的路径搜索
            paths = find_all_paths(graph, entry, exit_, max_paths=500, cutoff=80)
            if not paths:
                print(f"[INFO] No CFG paths found for {code_file}, skip.")
                continue

            save_paths_meta_compressed(
                nodes_file, edges_file, paths,
                output_root=output_dir,
                real_label=real_label
            )
        except Exception as e:
            # 打印更详细的错误信息
            print(f"[ERROR] Failed processing {code_file}: {repr(e)}")
            continue


# 主程序
def main():
    code_file_dir   = '/root/AMB/MAGNET-main/one_step_run/dateset/devign_raw_code2'
    parsed_file_dir = '/root/AMB/MAGNET-main/pre_process_dataset/devign_parsed'
    magnet_json     = '/root/AMB/MAGNET-main/one_step_run/dateset/Magnet_devign.json'
    output_dir      = '/root/AMB/TEST_TRY/code/output_compressed1'

    # None 表示“全量跑”
    num_samples     = None
    
    process_sample_files(code_file_dir, parsed_file_dir, magnet_json, output_dir, num_samples)


if __name__ == "__main__":
    main()
