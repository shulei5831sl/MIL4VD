import os
import json
import csv
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch


class PathMILDataset(Dataset):
    """
    基于 *_paths.json 的函数级 MIL 数据集：
    - 每个 json 文件 = 一个函数（bag）
    - json 里有多条 basic path，每条 basic path -> 一个 PyG Data 图
    """

    def __init__(self, json_root: str):
        """
        :param json_root: 你的 output_compressed 目录
        """
        super().__init__()
        self.json_root = json_root

        # 1) 找到所有 *_paths.json
        self.meta_files: List[str] = sorted(
            os.path.join(json_root, f)
            for f in os.listdir(json_root)
            if f.endswith("_paths.json")
        )
        if not self.meta_files:
            print(f"[WARN] 在目录 {json_root} 下没有找到 *_paths.json")

        # 2) 构建 sample 列表 + type vocab
        self.samples: List[Dict[str, Any]] = []
        self.type2id: Dict[str, int] = {}

        self._build_samples_and_vocab()

        # 最终节点特征维度：
        #   one-hot(type) + isCFGNode(1维) + isPrefix(1维)
        self.num_node_features: int = len(self.type2id) + 2

    # ---------- 内部工具 ----------

    @staticmethod
    def _load_nodes(nodes_file: str) -> Dict[int, Dict[str, Any]]:
        nodes = {}
        with open(nodes_file, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                nid = int(row["key"])
                nodes[nid] = {
                    "type": row["type"],
                    "isCFGNode": (row.get("isCFGNode", "False") == "True"),
                    # 其它字段先留着，有需要再用
                    "code": row.get("code", ""),
                    "location": row.get("location", ""),
                    "functionId": row.get("functionId", ""),
                    "operator": row.get("operator", ""),
                    "baseType": row.get("baseType", ""),
                    "completeType": row.get("completeType", ""),
                    "identifier": row.get("identifier", ""),
                }
        return nodes

    @staticmethod
    def _load_all_edges(edges_file: str) -> List[Dict[str, Any]]:
        """
        不筛选 type，先把所有边读出来，后面在子图里再根据节点集合过滤。
        """
        edges = []
        with open(edges_file, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    s = int(row["start"])
                    e = int(row["end"])
                except ValueError:
                    continue
                edges.append(
                    {
                        "start": s,
                        "end": e,
                        "type": row.get("type", ""),
                        "var": row.get("var", ""),
                    }
                )
        return edges

    def _build_samples_and_vocab(self):
        """
        - 构造 self.samples：每个 sample 对应一个函数
        - 顺便扫一遍 nodes.csv，统计所有 node.type，建立 type2id
        """
        type_set = set()

        for meta_path in self.meta_files:
            with open(meta_path, "r") as f:
                meta = json.load(f)

            cpg_dir = meta["cpg_dir"]
            nodes_file = meta["nodes_file"]
            edges_file = meta["edges_file"]
            real_label = int(meta["real_label"])
            common_prefix = meta.get("common_prefix", [])
            path_items = meta.get("paths", [])

            # 还原每条路径的完整节点序列：prefix + suffix
            paths_nodes: List[List[int]] = []
            for p in path_items:
                suffix = p.get("suffix", [])
                node_seq = list(common_prefix) + list(suffix)
                if len(node_seq) == 0:
                    continue
                paths_nodes.append(node_seq)

            if len(paths_nodes) == 0:
                # 没有路径的函数，可以选择跳过
                print(f"[INFO] {meta_path} 没有有效路径，跳过。")
                continue

            # ⭐ 把 prefix 的长度也存起来，后面构造 is_prefix 特征要用
            prefix_len = len(common_prefix)

            self.samples.append(
                {
                    "cpg_dir": cpg_dir,
                    "nodes_file": nodes_file,
                    "edges_file": edges_file,
                    "real_label": real_label,
                    "paths_nodes": paths_nodes,
                    "prefix_len": prefix_len,
                }
            )

            # --- 顺便扫 nodes_file，收集 node.type ---
            if os.path.exists(nodes_file):
                with open(nodes_file, "r") as nf:
                    reader = csv.DictReader(nf, delimiter="\t")
                    for row in reader:
                        t = row.get("type", "")
                        if t:
                            type_set.add(t)
            else:
                print(f"[WARN] nodes_file 不存在: {nodes_file}")

        # 建立 type -> id
        self.type2id = {t: idx for idx, t in enumerate(sorted(type_set))}
        print(f"[INFO] 构建完数据集，共有 {len(self.samples)} 个函数样本（bag），"
              f"节点类型数={len(self.type2id)}")

    # ---------- Dataset 接口 ----------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        返回：
        {
          "data_list": [Data(path1), Data(path2), ...],
          "bag_label": 0/1,
          "cpg_dir": xxx
        }
        """
        sample = self.samples[idx]
        nodes_file = sample["nodes_file"]
        edges_file = sample["edges_file"]
        real_label = sample["real_label"]
        paths_nodes = sample["paths_nodes"]
        prefix_len = sample["prefix_len"]  # ⭐ 函数级的公共前缀长度

        # 加载 nodes / edges
        nodes = self._load_nodes(nodes_file)
        edges = self._load_all_edges(edges_file)

        data_list: List[Data] = []

        for path_node_seq in paths_nodes:
            # 1) 这个 basic path 涉及到的节点集合（去重后按出现顺序）
            seen = []
            for nid in path_node_seq:
                if nid in nodes and nid not in seen:
                    seen.append(nid)

            if len(seen) == 0:
                continue

            # 这里简单认为：前 prefix_len 个节点属于前缀部分
            # 为安全起见做个裁剪
            eff_prefix_len = min(prefix_len, len(seen))
            is_prefix_flags = [1 if i < eff_prefix_len else 0 for i in range(len(seen))]

            # 2) old_nid -> new_idx 映射
            nid2idx = {nid: i for i, nid in enumerate(seen)}

            # 3) 构造节点特征 x：[num_nodes, feature_dim]
            #    feature = [ one-hot(type) , isCFGNode , isPrefix ]
            feat_dim = self.num_node_features
            x_list = []
            for i, nid in enumerate(seen):
                info = nodes[nid]
                feat = torch.zeros(feat_dim, dtype=torch.float)

                t = info["type"]
                if t in self.type2id:
                    feat[self.type2id[t]] = 1.0

                # 倒数第二维：isCFGNode
                feat[-2] = 1.0 if info["isCFGNode"] else 0.0
                # 最后一维：是否来自公共前缀
                feat[-1] = float(is_prefix_flags[i])

                x_list.append(feat)

            x = torch.stack(x_list, dim=0)  # [N, F]

            # 4) 构造子图的 edge_index
            edge_index_list = []
            for e in edges:
                s = e["start"]
                d = e["end"]
                if s in nid2idx and d in nid2idx:
                    edge_index_list.append([nid2idx[s], nid2idx[d]])

            if len(edge_index_list) == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

            data = Data(x=x, edge_index=edge_index)
            data_list.append(data)

        if len(data_list) == 0:
            # 极端情况，造一个最小 dummy 图防止崩
            dummy_x = torch.zeros((1, self.num_node_features), dtype=torch.float)
            dummy_data = Data(x=dummy_x, edge_index=torch.empty((2, 0), dtype=torch.long))
            data_list.append(dummy_data)

        return {
            "data_list": data_list,
            "bag_label": int(real_label),
            "cpg_dir": sample["cpg_dir"],
        }


def path_mil_collate_fn(samples: List[Dict[str, Any]]) -> Batch:
    """
    collate_fn:
      - 把一个 batch 里的若干函数（bag）打平成一个 Batch
      - 添加 batch.path2bag, batch.bag_labels，供 MIL loss 使用
    """
    all_data = []
    path2bag = []
    bag_labels = []

    for bag_id, sample in enumerate(samples):
        bag_labels.append(sample["bag_label"])
        for d in sample["data_list"]:
            # 记录路径属于哪一个 bag
            d.bag_id = bag_id
            all_data.append(d)
            path2bag.append(bag_id)

    if len(all_data) == 0:
        # 极端情况，造一个空 batch
        return None

    batch = Batch.from_data_list(all_data)
    batch.path2bag = torch.tensor(path2bag, dtype=torch.long)
    batch.bag_labels = torch.tensor(bag_labels, dtype=torch.float)
    return batch
