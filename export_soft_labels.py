# export_soft_labels.py

import os
import json
from types import SimpleNamespace

import torch

from dataset_path_mil import PathMILDataset
from dataset_path_mil import path_mil_collate_fn
from path_encoder import PathMILModel
from tqdm import tqdm
import numpy as np

def export_soft_labels(
    data_root: str,
    ckpt_path: str,
    # topk_ratio: float = 0.2
):
    # 1) 加载数据集（和训练时一样）
    dataset = PathMILDataset(data_root)

    args = SimpleNamespace()
    args.num_node_features    = dataset.num_node_features
    args.node_hidden_dim      = 64
    args.num_layers           = 2
    args.poollist             = [10]
    args.hierarchical_pool_rate = [0.5]
    args.pool_act             = 'relu'
    args.pool_drop            = 0.0
    args.pool_pool_type       = 'topk'
    args.conv_type            = 'gcn'
    args.GNNact               = 'relu'
    args.GNNdrop              = 0.1
    args.num_heads            = 4
    args.GTnorm               = 'ln'
    args.GTdrop               = 0.1
    args.attn_drop            = 0.1
    args.middle_layer_type    = 'bn'
    args.skip_connection      = 'short'
    args.readout              = 'none'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PathMILModel(args).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    print(f"[INFO] 已加载模型：{ckpt_path}")
    print(f"[INFO] 数据集共有 {len(dataset)} 个函数（bag）")

    # for idx in range(len(dataset)):
    for idx in tqdm(range(len(dataset)), desc = "export soft labels"):
        sample = dataset[idx]  # {"data_list": [...], "bag_label": 0/1, "cpg_dir": xxx}

        # 构造单个 bag 的 batch
        batch = path_mil_collate_fn([sample])
        batch = batch.to(device)

        with torch.no_grad():
            path_logits = model(batch)                # [num_paths]
            path_probs = torch.sigmoid(path_logits)   # [num_paths]
            path_probs = path_probs.detach().cpu().numpy()

        bag_label = int(batch.bag_labels.item())
        num_paths = len(sample["data_list"])

        
    
        # if bag_label == 1:
            # 函数有漏洞：用 Top-k 把最可疑的若干条路径标成 1
            # k = max(1, int(num_paths * topk_ratio))
            # sorted_idx = path_probs.argsort()[::-1]  # 概率从大到小排序
            # top_idx = sorted_idx[:k]
            # for j in top_idx:
            #     soft_labels[int(j)] = 1.0
        # 默认软标签全 0
        # 默认软标签全 0
        soft_labels = [0.0] * num_paths
        if bag_label == 1:
            p = path_probs.astype(np.float64)
            p = np.clip(p, 1e-6, 1.0 - 1e-6)
            soft_labels = p.tolist()
        else:
            soft_labels = [0.0] * num_paths



        # 找到对应的 *_paths.json
        cpg_dir = sample["cpg_dir"]  # 例如 "0bdab813e1002bc8....c"
        json_path = os.path.join(data_root, f"{cpg_dir}_paths.json")

        if not os.path.exists(json_path):
            print(f"[WARN] {json_path} 不存在，跳过。")
            continue

        with open(json_path, "r") as f:
            meta = json.load(f)

        if "paths" not in meta:
            print(f"[WARN] {json_path} 中没有 paths 字段，跳过。")
            continue

        if len(meta["paths"]) != num_paths:
            print(f"[WARN] {json_path} 中路径数不一致（meta={len(meta['paths'])}, 实际={num_paths}），尽量对齐前者。")

        # 只更新前 len(meta["paths"]) 个
        for i, p in enumerate(meta["paths"]):
            if i < len(soft_labels):
                p["soft_label"] = float(soft_labels[i])
            else:
                # 万一 meta 里比 data_list 多，就给剩下的补 0
                p["soft_label"] = 0.0

        with open(json_path, "w") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)

        # if (idx + 1) % 50 == 0 or idx == len(dataset) - 1:
        #     print(f"[INFO] 已处理 {idx + 1}/{len(dataset)} 个函数")

    print("[INFO] 软标签导出完成。")


if __name__ == "__main__":
    data_root = "/root/AMB/TEST_TRY/code/output_compressed1_2"
    ckpt_path = "/root/AMB/TEST_TRY/code/unixc/path_mil_best2.pt"

    export_soft_labels(
        data_root=data_root,
        ckpt_path=ckpt_path,
        # topk_ratio=0.2,   # 你可以自己调 0.1 / 0.2 / 0.3 等
    )
