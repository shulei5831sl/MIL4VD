# path_level_dataset.py
import os
import glob
import json
import csv
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


def _load_nodes(nodes_file: str) -> Dict[int, dict]:
    """把 nodes.csv 读成 {node_id: row_dict}"""
    nodes = {}
    with open(nodes_file, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            nid = int(row["key"])
            nodes[nid] = row
    return nodes


def _load_cfg_edges(edges_file: str):
    """只保留 FLOWS_TO 边"""
    edges = []
    with open(edges_file, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["type"] != "FLOWS_TO":
                continue
            edges.append((int(row["start"]), int(row["end"])))
    return edges


class PathLevelDataset(Dataset):
    """
    每个样本 = 一条 basic path 的子图：
      - x: [num_nodes, num_node_types] one-hot（根据 node['type']）
      - edge_index: [2, num_edges]（仅 FLOWS_TO，限制在该路径节点上）
      - y: 0/1（路径软标签）
    """

    def __init__(self, meta_files: List[str], node_type_vocab: Dict[str, int] = None):
        """
        meta_files: 一批 *.c_paths.json 路径
        node_type_vocab: 可选；如果传入则复用，不传则在这些 meta_files 上重新统计
        """
        self.meta_files = meta_files
        self.node_type_vocab = node_type_vocab or {}
        self._build_vocab_if_needed()

        self.num_node_types = len(self.node_type_vocab)

        # 缓存 nodes / edges，避免重复读文件
        self.nodes_cache = {}
        self.edges_cache = {}

        # 真正的样本列表
        self.samples = []
        self._build_samples()

        print(f"[PathLevelDataset] 共 {len(self.samples)} 条路径样本，"
              f"节点类型数 = {self.num_node_types}")

    def _build_vocab_if_needed(self):
        """如果没给 vocab，就在所有 nodes_file 里把 node['type'] 扫一遍"""
        if self.node_type_vocab:
            return
        vocab = {}
        idx = 0
        for meta_path in self.meta_files:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            nodes_file = meta["nodes_file"]
            if not os.path.exists(nodes_file):
                continue
            with open(nodes_file, "r") as f_nodes:
                reader = csv.DictReader(f_nodes, delimiter="\t")
                for row in reader:
                    t = row["type"]
                    if t not in vocab:
                        vocab[t] = idx
                        idx += 1
        self.node_type_vocab = vocab

    def _build_samples(self):
        """把 meta 展开为一个个 path 样本"""
        func_idx = 0

        for meta_path in self.meta_files:
            with open(meta_path, "r") as f:
                meta = json.load(f)

            nodes_file = meta["nodes_file"]
            edges_file = meta["edges_file"]
            func_label = int(meta["real_label"])
            common_prefix = meta.get("common_prefix", [])
            func_name = meta.get("cpg_dir","")

            seen_paths = set()

            for p in meta["paths"]:
                path_id = p["path_id"]
                suffix = p["suffix"]
                # soft_label = int(p["soft_label"])
                # soft_raw = p["soft_label"]
                # soft_label = int(soft_raw > 0.5)
                soft_raw = float(p.get("soft_label", 0.0))
                soft_label = soft_raw
                node_ids = common_prefix + suffix
                if not node_ids:
                    continue
                
                key = tuple(node_ids)
                if key in seen_paths:
                    continue
                seen_paths.add(key)

                self.samples.append(
                    {
                        "nodes_file": nodes_file,
                        "prefix_len": len(common_prefix),
                        "edges_file": edges_file,
                        "node_ids": node_ids,
                        "soft_label": soft_label,
                        "func_label": func_label,
                        "func_name": func_name,
                        #"func_name": meta.get("cpg_dir", ""),
                        "path_id": path_id,
                        "func_id": func_idx,
                        "soft_raw": float(soft_raw),
                    }
                )
                
            func_idx += 1

    def __len__(self):
        return len(self.samples)

    def _get_nodes(self, nodes_file):
        if nodes_file not in self.nodes_cache:
            self.nodes_cache[nodes_file] = _load_nodes(nodes_file)
        return self.nodes_cache[nodes_file]

    def _get_edges(self, edges_file):
        if edges_file not in self.edges_cache:
            self.edges_cache[edges_file] = _load_cfg_edges(edges_file)
        return self.edges_cache[edges_file]

    def __getitem__(self, idx):
        s = self.samples[idx]
        nodes_file = s["nodes_file"]
        edges_file = s["edges_file"]
        node_ids = s["node_ids"]
        #y = s["soft_label"] # 只参与训练用
        y_soft = float(s["soft_label"])   # 软标签
        soft_raw = s["soft_raw"] 

        prefix_len = s.get("prefix_len", 0) # 是否是后缀，路径支线
        # 读取完整 nodes / edges
        nodes_dict = self._get_nodes(nodes_file)
        edges_list = self._get_edges(edges_file)

        # 节点 id 映射到 [0 .. n-1]
        id2local = {nid: i for i, nid in enumerate(node_ids)}
        num_nodes = len(node_ids)

        feat_dim = self.num_node_types + 1  # +1 给 is_prefix，增加一个维度，给判断是否是子路径
        # x: one-hot(node_type)
        x = torch.zeros((num_nodes, feat_dim), dtype=torch.float)
        for i, nid in enumerate(node_ids):
            node = nodes_dict.get(nid)
            if node is None:
                continue
            t = node["type"]
            tid = self.node_type_vocab.get(t)
            if tid is not None:
                x[i, tid] = 1.0

        if i< prefix_len:
            x[i, -1] = 1.0
        else:
            x[i, -1] = 0.0
        # edge_index: 只保留落在这条 path 上的 FLOWS_TO
        edge_index_list = []
        for u, v in edges_list:
            if u in id2local and v in id2local:
                edge_index_list.append([id2local[u], id2local[v]])

        if len(edge_index_list) == 0:
            # 至少给个自环，防止空图
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

        y = torch.tensor([y_soft], dtype=torch.float)
 
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
        )

        # 可选：附带函数级标签，后面做分析用
        data.soft_raw = torch.tensor(soft_raw, dtype=torch.float)
        data.func_id   = torch.tensor(s["func_id"], dtype=torch.long)
        data.path_id   = torch.tensor(s["path_id"], dtype=torch.long)
        data.func_label = torch.tensor(s["func_label"], dtype=torch.long)
        data.func_name = s["func_name"]
        data.sample_idx = torch.tensor(idx, dtype=torch.long)
        # data.func_id = torch.tensor(s["func_id"], dtype=torch.long)
        # data.path_id = s["path_id"]


        return data
