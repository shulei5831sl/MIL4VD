import torch
from utils import debug
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
# dafen.py 里
from train_path_level import recover_path_code  # 直接复用

@torch.no_grad()
def debug_one_function(dataloader, device, model,
                       target_func_id=48,
                       topk=3,
                       max_nodes=15,
                       min_prob=None):
    model.eval()
    path_infos = []

    for batch in dataloader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, None, batch.batch)
        probs = F.softmax(logits, dim=-1)[:, 1]

        func_ids = batch.func_id.cpu().numpy()
        path_ids = batch.path_id.cpu().numpy()
        soft_raws = batch.soft_raw.cpu().numpy()
        sample_indices = batch.sample_idx.cpu().numpy()

        for p, fid, pid, sraw, sidx in zip(
            probs.cpu().numpy(), func_ids, path_ids, soft_raws, sample_indices
        ):
            if int(fid) != int(target_func_id):
                continue
            if (min_prob is not None) and (p < min_prob):
                continue
            path_infos.append({
                "prob": float(p),
                "path_id": int(pid),
                "soft_raw": float(sraw),
                "sample_idx": int(sidx),
            })

    if not path_infos:
        print(f"[FUNC {target_func_id}] 没有路径满足条件（min_prob={min_prob}）")
        return

    path_infos.sort(key=lambda x: x["prob"], reverse=True)

    print(f"[FUNC {target_func_id}] 收集到 {len(path_infos)} 条路径"
          + (f"（min_prob={min_prob}）" if min_prob is not None else ""))
    for rank, info in enumerate(path_infos[:topk]):
        print(f"  #{rank:02d} path_prob={info['prob']:.4f}, "
              f"path_id={info['path_id']}, soft_raw={info['soft_raw']:.4f}, "
              f"sample_idx={info['sample_idx']}")

        code_lines = recover_path_code(dataloader.dataset,
                                       info["sample_idx"])
        for line in code_lines[:max_nodes]:
            print("       " + line)


# def debug_one_function(dataloader, device, model, target_func_id, topk=5):
#     model.eval()
#     with torch.no_grad():
#         for data in dataloader:
#             data = data.to(device)
#             edge_attr = getattr(data, "edge_attr", None)

#             logits = model(data.x, data.edge_index, edge_attr, data.batch)
#             probs = F.softmax(logits, dim=-1)[:, 1]  # [B]
#             func_ids = data.func_id
#             func_labs = data.func_label

#             for p, fid, flab in zip(probs, func_ids, func_labs):
#                 fid = int(fid.item())
#                 if fid == target_func_id:
#                     debug(
#                         f"[FUNC {fid}] label={int(flab.item())}", 
#                         f"path_prob={float(p.item()):.4f}"
#                     )
#     model.train()