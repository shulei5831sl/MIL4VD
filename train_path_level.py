# train_path_level.py
import os
import glob
import random
from types import SimpleNamespace
from collections import defaultdict
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)

from path_level_dataset import PathLevelDataset
from path_classifier import PathClassifier
from utils import debug

# 这些只是默认值，真正用的时候我在调用 evaluate_function_level 时会显式传
PATH_PROB_MIN = 0.05
FUNC_THRESHOLD = 0.65


# ===== 工具：聚合若干路径得函数概率 =====
# def aggregate_scores(scores, mode="noisy_or", topk=5, min_p=0.4):
#     """
#     scores: list[float]，每个是 p(path=1)
#     mode:
#       - "noisy_or": 1 - Π(1 - s_i)
#       - "max": max(s_i)
#     """
#     if not scores:
#         return 0.0

#     scores_sorted = sorted(scores, reverse=True)
#     scores_filtered = [s for s in scores_sorted if s >= min_p]
#     if not scores_filtered:
#         return 0.0

#     if topk is not None and len(scores_filtered) > topk:
#         scores_use = scores_filtered[:topk]
#     else:
#         scores_use = scores_filtered

#     if not scores_use:
#         return 0.0

#     if mode == "noisy_or":
#         q = 1.0
#         for s in scores_use:
#             q *= (1.0 - s)
#         prob = 1.0 - q
#     elif mode == "max":
#         prob = max(scores_use)
#     else:
#         raise NotImplementedError(f"Unknown mode={mode}")

#     return float(prob)
def aggregate_scores(scores, mode="noisy_or", topk=3, min_frac=0.5):
    """
    scores: list[float]
    mode : "noisy_or" / "max"
    topk : 最多看多少条路径
    min_frac: 只保留 >= min_frac * p_max 的路径
    """
    if not scores:
        return 0.0

    scores_sorted = sorted(scores, reverse=True)
    p_max = scores_sorted[0]

    # 1) 相对阈值：至少保留那些“接近最大值”的路径
    thresh = min_frac * p_max
    cand = [s for s in scores_sorted if s >= thresh]

    # 2) 保底：如果全被砍光了，至少留一条 top-1
    if not cand:
        cand = [p_max]

    # 3) 截断 topk
    if topk is not None and len(cand) > topk:
        scores_use = cand[:topk]
    else:
        scores_use = cand

    if mode == "noisy_or":
        q = 1.0
        for s in scores_use:
            q *= (1.0 - s)
        prob = 1.0 - q
    elif mode == "max":
        prob = max(scores_use)
    else:
        raise NotImplementedError(f"Unknown mode={mode}")

    return float(prob)


def aggregate_noisy_or_topk(scores, topk=None, min_p=None):
    """
    保留一个更简洁版 noisy-OR，备用
    """
    if min_p is not None:
        scores = [s for s in scores if s >= min_p]
    if not scores:
        return 0.0

    scores_sorted = sorted(scores, reverse=True)
    if topk is not None and len(scores_sorted) > topk:
        scores_use = scores_sorted[:topk]
    else:
        scores_use = scores_sorted

    q = 1.0
    for s in scores_use:
        q *= (1.0 - s)
    return 1.0 - q


# ===== 把一个路径 sample_idx 对应的代码恢复出来，用于 debug =====
def recover_path_code(dataset, sample_idx: int,
                      max_prefix: int = 3,
                      max_suffix: int = 15):
    if sample_idx < 0 or sample_idx >= len(dataset.samples):
        return [f"[WARN] sample_idx={sample_idx} out of range"]

    s = dataset.samples[sample_idx]
    nodes_file = s["nodes_file"]
    node_ids   = s["node_ids"]
    prefix_len = s.get("prefix_len", 0)

    nodes_dict = dataset._get_nodes(nodes_file)

    lines = []

    # 公共前缀部分
    prefix_count = 0
    for nid in node_ids[:prefix_len]:
        node = nodes_dict.get(nid)
        if node is None:
            continue
        code = (node.get("code") or node.get("CODE") or "").strip()
        if not code:
            continue
        lines.append(f"[P nid={nid}] {code}")
        prefix_count += 1
        if prefix_count >= max_prefix:
            break

    # 后缀部分
    suffix_count = 0
    for nid in node_ids[prefix_len:]:
        node = nodes_dict.get(nid)
        if node is None:
            continue
        code = (node.get("code") or node.get("CODE") or "").strip()
        if not code:
            continue
        lines.append(f"[S nid={nid}] {code}")
        suffix_count += 1
        if suffix_count >= max_suffix:
            break

    if not lines:
        lines.append("[EMPTY PATH CODE]")

    return lines


# ===== 函数级 debug：看每个函数下 top-k 路径 =====
@torch.no_grad()
def debug_func_paths(model, dataset, device, max_funcs=3,
                     agg_mode="noisy_or", topk=3,
                     path_prob_min=PATH_PROB_MIN,
                     threshold=0.6):
    from torch_geometric.loader import DataLoader as _DL

    loader = _DL(dataset, batch_size=64, shuffle=False)
    model.eval()

    func_paths = defaultdict(list)
    func_label_map = {}
    func_name_map  = {}

    for batch in loader:
        batch = batch.to(device)
        edge_attr = getattr(batch, "edge_attr", None)

        logits = model(batch.x, batch.edge_index, edge_attr, batch.batch)
        # ★ 这里改成 sigmoid(logit_pos)
        pos_logits = logits[:, 1]
        probs = torch.sigmoid(pos_logits)

        func_ids       = batch.func_id.cpu().numpy()
        func_labels    = batch.func_label.cpu().numpy()
        path_ids       = batch.path_id.cpu().numpy()
        soft_raws      = batch.soft_raw.cpu().numpy()
        sample_indices = batch.sample_idx.cpu().numpy()
        probs_np       = probs.cpu().numpy()

        has_func_name = hasattr(batch, "func_name")

        for i, (p, fid, flab, pid, sraw, sidx) in enumerate(
            zip(probs_np, func_ids, func_labels, path_ids, soft_raws, sample_indices)
        ):
            fid  = int(fid)
            flab = int(flab)

            func_paths[fid].append({
                "prob": float(p),
                "soft_raw": float(sraw),
                "path_id": int(pid),
                "sample_idx": int(sidx),
            })
            func_label_map[fid] = flab

            if has_func_name:
                raw_name = batch.func_name[i]
                short_name = os.path.basename(raw_name) if raw_name else f"func_{fid}"
            else:
                short_name = f"func_{fid}"
            func_name_map[fid] = short_name

    count = 0
    for fid, paths in func_paths.items():
        flab = func_label_map[fid]
        if flab != 1:
            continue
        count += 1
        if count > max_funcs:
            break

        paths_sorted = sorted(paths, key=lambda x: x["prob"], reverse=True)
        scores = [p["prob"] for p in paths_sorted]

        # p_func = aggregate_scores(
        #     scores,
        #     mode=agg_mode,
        #     topk=topk,
        #     min_p=path_prob_min
        # )

        p_func = aggregate_scores(
            scores,
            mode=agg_mode,
            topk=topk,
            min_frac=0.5   # 这里先写死 0.5，相当于“保留 >= 0.5 * max 的路径”
        )


        name_display = func_name_map.get(fid, f"func_{fid}")
        print("=" * 80)
        print(f"FuncID={fid}, name={name_display}, real_label={flab}")
        print(f"  aggregated p(func=1)={p_func:.4f}, "
              f"threshold={threshold}, pred={int(p_func >= threshold)}")
        print(f"  path num = {len(paths_sorted)}")

        for i, info in enumerate(paths_sorted[:2]):
            print(f"    #{i:02d} path_id={info['path_id']}, "
                  f"prob={info['prob']:.4f}, soft_raw={info['soft_raw']:.4f}")

            code_lines = recover_path_code(
                dataset,
                info["sample_idx"],
                max_prefix=3,
                max_suffix=8
            )
            for line in code_lines:
                print(f"          {line}")


# ===== 函数级评估（BCE + 软标签 + noisy-OR 聚合） =====
@torch.no_grad()
def evaluate_function_level(dataloader, device, model, criterion,
                            agg_mode="noisy_or", topk=3,
                            path_prob_min=PATH_PROB_MIN,
                            func_threshold=FUNC_THRESHOLD,
                            do_sweep=False):
    """
    criterion: BCEWithLogitsLoss 实例
    返回：
      avg_loss, acc, pr, rc, f1, best_th
      - 当 do_sweep=False 时，best_th 就是传进来的 func_threshold
      - 当 do_sweep=True 时，best_th 是在当前 dataloader 上 F1 最高的阈值
    """
    model.eval()
    func_probs = defaultdict(list)
    func_labels = {}
    all_loss = []

    # 先把每条路径的 prob 聚合到函数上
    for data in dataloader:
        data = data.to(device)
        edge_attr = getattr(data, "edge_attr", None)

        logits = model(data.x, data.edge_index, edge_attr, data.batch)
        pos_logits = logits[:, 1]          # [N]
        targets = data.y.view(-1)          # [N] float in [0,1]

        loss = criterion(pos_logits, targets)
        all_loss.append(loss.detach().cpu().item())

        probs = torch.sigmoid(pos_logits).detach().cpu().numpy()
        func_ids = data.func_id.detach().cpu().numpy()
        func_labs = data.func_label.detach().cpu().numpy()

        for p, fid, flab in zip(probs, func_ids, func_labs):
            fid = int(fid)
            func_probs[fid].append(float(p))
            func_labels[fid] = int(flab)

    # 函数级 label & 聚合后的概率
    y_true = []
    func_probs_list = []
    for fid, scores in func_probs.items():
        label = func_labels[fid]

        prob = aggregate_scores(scores, mode=agg_mode, topk=topk, min_frac=0.5)

        y_true.append(label)
        func_probs_list.append(prob)

    avg_loss = float(np.mean(all_loss)) if all_loss else 0.0
    y_true_arr = np.array(y_true)
    probs_arr  = np.array(func_probs_list)

    # 小工具：给定一个阈值 th，算一遍各指标
    def compute_metrics_for_threshold(th):
        y_pred_th = (probs_arr >= th).astype(int)
        acc_th = accuracy_score(y_true_arr, y_pred_th) * 100.0
        pr_th  = precision_score(y_true_arr, y_pred_th, zero_division=0) * 100.0
        rc_th  = recall_score(y_true_arr, y_pred_th, zero_division=0) * 100.0
        f1_th  = f1_score(y_true_arr, y_pred_th, zero_division=0) * 100.0
        cm_th  = confusion_matrix(y_true_arr, y_pred_th)
        return acc_th, pr_th, rc_th, f1_th, cm_th

    # 先用当前 func_threshold 算一遍
    acc, pr, rc, f1, cm = compute_metrics_for_threshold(func_threshold)
    best_th = func_threshold

    # 如需 sweep，就在一串阈值上找 F1 最高的那一个
    if do_sweep:
        thresholds = np.linspace(0.3, 0.8, 26)  # 0.30 ~ 0.80，每 0.02 一格
        best_f1_val = f1
        best_acc, best_pr, best_rc, best_cm = acc, pr, rc, cm

        for th in thresholds:
            acc_th, pr_th, rc_th, f1_th, cm_th = compute_metrics_for_threshold(th)
            if f1_th > best_f1_val:
                best_f1_val = f1_th
                best_th = th
                best_acc, best_pr, best_rc, best_cm = acc_th, pr_th, rc_th, cm_th

        acc, pr, rc, f1, cm = best_acc, best_pr, best_rc, best_f1_val, best_cm
        print(f"[SWEEP] best_func_threshold={best_th:.3f}, best_f1={best_f1_val:.2f}")

    print(cm)

    model.train()
    return avg_loss, acc, pr, rc, f1, best_th




# （可选）路径级评估：现在基本不用，你要也可以留着
@torch.no_grad()
def evaluate(dataloader, device, model, criterion, path_threshold=0.5):
    model.eval()
    all_targets, all_probs, all_loss = [], [], []

    for data in dataloader:
        data = data.to(device)
        edge_attr = getattr(data, "edge_attr", None)

        logits = model(data.x, data.edge_index, edge_attr, data.batch)
        pos_logits = logits[:, 1]
        targets = data.y.view(-1)

        loss = criterion(pos_logits, targets)
        all_loss.append(loss.detach().cpu().item())

        probs = torch.sigmoid(pos_logits).detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_targets.extend(targets.detach().cpu().numpy().tolist())

    avg_loss = float(np.mean(all_loss)) if all_loss else 0.0

    y_true = (np.array(all_targets) >= 0.5).astype(int)
    y_pred = (np.array(all_probs) >= path_threshold).astype(int)

    acc = accuracy_score(y_true, y_pred) * 100.0
    pr  = precision_score(y_true, y_pred, zero_division=0) * 100.0
    rc  = recall_score(y_true, y_pred, zero_division=0) * 100.0
    f1  = f1_score(y_true, y_pred, zero_division=0) * 100.0
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    model.train()
    return avg_loss, acc, pr, rc, f1


# ===== 训练循环：用 BCE + 软标签 =====
def train(args, train_loader, val_loader, test_loader, device, model, criterion):
    global FUNC_THRESHOLD
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_iters
        )
    elif args.scheduler == "linear":
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=args.max_iters
        )
    else:
        scheduler = None

    from tqdm import tqdm
    train_loader_iter = iter(train_loader)
    total_loss = 0.0

    best_val_f1 = -1.0
    best_state = None
    patience = 0
    max_patience = getattr(args, "max_patience", 10)

    # 为了在 log 里显示 test_*，提前放几个占位
    last_test_acc = 0.0
    last_test_f1  = 0.0

    total_steps = args.max_iters
    pbar = tqdm(range(1, total_steps + 1))
    interval_start = time.time()

    for step in pbar:
        model.train()
        try:
            data = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            data = next(train_loader_iter)

        data = data.to(device)
        edge_attr = getattr(data, "edge_attr", None)

        logits = model(data.x, data.edge_index, edge_attr, data.batch)
        pos_logits = logits[:, 1]       # ★ 只看正类 logit
        targets = data.y.view(-1)       # ★ 软标签 float

        loss = criterion(pos_logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        pbar.set_description(f"Train step {step}")
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if step % args.eval_step == 0:
            interval_time = time.time() - interval_start

            train_loss = total_loss / float(args.eval_step)
            total_loss = 0.0

            # 1) 训练集：不 sweep，用当前 FUNC_THRESHOLD
            train_loss_eval, train_acc, train_pr, train_rc, train_f1, _ = \
                evaluate_function_level(
                    train_loader, device, model, criterion,
                    agg_mode="noisy_or",
                    topk=3,
                    path_prob_min=PATH_PROB_MIN,
                    func_threshold=FUNC_THRESHOLD,
                    do_sweep=False
                )

            # 2) 验证集：sweep 找到最优阈值 + 最优 f1
            val_loss, val_acc, val_pr, val_rc, val_f1, best_th = \
                evaluate_function_level(
                    val_loader, device, model, criterion,
                    agg_mode="noisy_or",
                    topk=5,
                    path_prob_min=PATH_PROB_MIN,
                    func_threshold=FUNC_THRESHOLD,
                    do_sweep=True
                )

            # ★ 用验证集找到的 best_th 更新全局阈值
            FUNC_THRESHOLD = float(best_th)

            # 3) 测试集：用新的 FUNC_THRESHOLD 看效果
            test_loss, test_acc, test_pr, test_rc, test_f1, _ = \
                evaluate_function_level(
                    test_loader, device, model, criterion,
                    agg_mode="noisy_or",
                    topk=3,
                    path_prob_min=PATH_PROB_MIN,
                    func_threshold=FUNC_THRESHOLD,
                    do_sweep=False
                )

            print(
                f"[Iter {step}/{args.max_iters}] "
                f"train_loss={train_loss:.4f} "
                f"| val_loss={val_loss:.4f}, val_acc={val_acc:.2f}, val_f1={val_f1:.2f} "
                f"| test_acc={test_acc:.2f}, test_f1={test_f1:.2f} "
                f"| patience={patience:.2f} "
                f"| interval_time={interval_time:.1f}s({args.eval_step} steps)"
            )


            interval_start = time.time()

    # 还原最优参数，再在 test 上测一次最终指标
    if best_state is not None:
        model.load_state_dict(best_state)

    final_loss, final_acc, final_pr, final_rc, final_f1, _ = \
    evaluate_function_level(
        test_loader, device, model, criterion,
        agg_mode="noisy_or",
        topk=3,
        path_prob_min=PATH_PROB_MIN,
        func_threshold=FUNC_THRESHOLD,
        do_sweep=False
    )

    print(
        f"[Final Test] loss={final_loss:.4f}, acc={final_acc:.2f}, "
        f"pr={final_pr:.2f}, rc={final_rc:.2f}, f1={final_f1:.2f}"
    )


# ===== 主函数：读 meta → 划分 → 构建数据集 → 训练 =====
def main():
    # 1) 收集所有路径 meta json
    meta_dir = "/root/AMB/TEST_TRY/code/output_compressed1_2"
    meta_files = sorted(glob.glob(os.path.join(meta_dir, "*.json")))
    assert meta_files, f"meta_dir 为空: {meta_dir}"

    random.seed(10)
    random.shuffle(meta_files)

    n = len(meta_files)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_files = meta_files[:n_train]
    val_files = meta_files[n_train:n_train + n_val]
    test_files = meta_files[n_train + n_val:]

    print(f"[Split] train_funcs={len(train_files)}, "
          f"val_funcs={len(val_files)}, test_funcs={len(test_files)}")

    # 2) 构建数据集（train 上建 vocab，val/test 复用）
    train_dataset = PathLevelDataset(train_files)
    vocab = train_dataset.node_type_vocab
    num_node_types = train_dataset.num_node_types

    val_dataset = PathLevelDataset(val_files, node_type_vocab=vocab)
    test_dataset = PathLevelDataset(test_files, node_type_vocab=vocab)

    feat_dim = num_node_types + 1
    print(f"[INFO] Path-level feature dim = {feat_dim} (type {num_node_types} + is_prefix 1)")

    # 3) DataLoader
    num_workers = 8
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             persistent_workers=True)

    # 4.1 —— PATH 级软标签分布（把 y>=0.5 当“正”粗略数一下）
    # all_y = [float(d.y.item()) for d in train_dataset]
    # all_y = np.array(all_y)

    # num_pos_path = int((all_y >= 0.5).sum())
    # num_neg_path = int((all_y < 0.5).sum())
    # print(f"[Train PATH label stats] pos={num_pos_path}, neg={num_neg_path}")

    # if num_pos_path == 0:
    #     pos_weight_val = 1.0
    #     raw_ratio = 1.0
    # else:
    #     raw_ratio = num_neg_path / max(num_pos_path, 1)
    #     pos_weight_val = raw_ratio ** 0.5

    # print(f"[PATH pos_weight(before sqrt)] raw={raw_ratio:.2f}")
    # print(f"[PATH pos_weight(after  sqrt)] used={pos_weight_val:.2f}")

    # pos_weight = torch.tensor([pos_weight_val], dtype=torch.float)

    # 4.2 —— FUNC 级真实标签分布：只用来观察
    func_label_map = {}
    for d in train_dataset:
        fid = int(d.func_id)
        flab = int(d.func_label)
        func_label_map[fid] = flab
    func_labels = list(func_label_map.values())
    num_pos_func = sum(1 for v in func_labels if v == 1)
    num_neg_func = sum(1 for v in func_labels if v == 0)
    print(f"[Train FUNC label stats] pos={num_pos_func}, neg={num_neg_func}")

    all_y = np.array([float(d.y.item()) for d in train_dataset])  # [N], 每个是 0~1

    # 把 y 当成 “正类质量”：y=0.8 相当于 0.8 个正例，y=0.2 相当于 0.2 个正例
    pos_mass = all_y.sum()                   # 所有路径上的“正类质量”总和
    neg_mass = len(all_y) - pos_mass         # 剩下当成负类质量

    print(f"[Train PATH mass stats] pos_mass={pos_mass:.1f}, neg_mass={neg_mass:.1f}")

            # ===== 关键：结合真实标签 + 软标签来算 raw_ratio =====
    if pos_mass <= 0 or num_pos_func == 0:
        # 极端情况防止除 0
        raw_ratio = 1.0
        path_ratio = 1.0
        func_ratio = 1.0
    else:
        # 路径级：基于软标签的“不平衡比”
        path_ratio = neg_mass / pos_mass

        # 函数级：基于真实标签的不平衡比
        func_ratio = num_neg_func / num_pos_func

        # α 控制“更信伪标签”还是“更信真实标签”
        alpha = 0.5   # 可以先用 0.5，后面再调
        raw_ratio = (1.0 - alpha) * func_ratio + alpha * path_ratio

    pos_weight_val = raw_ratio ** 0.5


    print(f"[PATH pos_weight(before sqrt)] raw={raw_ratio:.2f}")
    print(f"[PATH pos_weight(after  sqrt)] used={pos_weight_val:.2f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float).to(device)
    pos_weight = pos_weight.to(device)

    # 5) 构造 args（PathEncoder + 训练超参）
    args = SimpleNamespace()

    # PathEncoder 相关
    args.num_node_features = feat_dim
    args.node_hidden_dim = 64

    args.poollist = [10, 20]
    args.hierarchical_pool_rate = [0.5, 0.5]
    args.pool_pool_type = "topk"
    args.pool_act = "relu"
    args.pool_drop = 0.2

    # GNN + GT
    args.conv_type = "gcn"
    args.GNNact = "relu"
    args.GNNdrop = 0.1
    args.num_layers = 2
    args.GTnorm = "ln"
    args.GTdrop = 0.2
    args.num_heads = 4
    args.attn_drop = 0.1

    args.middle_layer_type = "bn"
    args.skip_connection = "short"
    args.readout = "mean"   # 你之前就是 mean，就保持

    # 训练超参数
    args.lr = 1e-3
    args.weight_decay = 1e-4
    args.scheduler = "cosine"
    args.max_iters = 2000
    args.eval_step = 200
    args.max_patience = 10

    # 6) 模型 & 损失
    model = PathClassifier(args).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 7) 训练
    train(args, train_loader, val_loader, test_loader, device, model, criterion)

    print(f"[Train FUNC label stats] pos={num_pos_func}, neg={num_neg_func}")
    print(f"[Train PATH label stats] pos={pos_mass}, neg={neg_mass}")
    print(f"[pos_weight] {pos_weight}")

    # 8) 训练完之后，随便 debug 看几个函数下的 top-k 路径
    debug_func_paths(
        model,
        test_dataset,
        device,
        max_funcs=3,
        agg_mode="noisy_or",
        topk=3,
        path_prob_min=PATH_PROB_MIN,
        threshold=0.6
    )

    # 你原来 dafen.py 里的 debug_one_function 也可以继续用
    from dafen import debug_one_function
    debug_one_function(
        test_loader,
        device,
        model,
        target_func_id=0,
        topk=3,
        max_nodes=15
    )


if __name__ == "__main__":
    main()
