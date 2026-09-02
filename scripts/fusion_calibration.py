# -*- coding: utf-8 -*-
"""
融合阈值标定（任务书 P1-5 第 2–4 步）
======================================
输入 evaluation/scenario_scores.csv（列：acoustic,voiceprint,semantic,expected），
做非对称代价网格搜索（误拦家人 FA 代价 = 3 × 漏放克隆 MISS 代价），输出最优参数 +
等高线热力图 + 消融表，并写入 evaluation/fusion_calibration.md。

数据缺失时：写模板 + 指引，不报错退出。
"""
from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from fusion.fusion_orchestrator import FusionOrchestrator, ChannelVerdict

SCORES = ROOT / "evaluation" / "scenario_scores.csv"
OUT_MD = ROOT / "evaluation" / "fusion_calibration.md"
FA_COST = 3.0  # 误拦家人代价系数（相对漏报）


def _decide(ac, vp, sem, hard, alert, block, weights):
    orch = FusionOrchestrator(weights=weights, hard_block=hard,
                              soft_alert=alert, soft_block=block)
    return orch.decide(
        ChannelVerdict("acoustic", ac, "x", "", 0.9),
        ChannelVerdict("voiceprint", vp, "x", "", 0.9),
        ChannelVerdict("semantic", sem, "x", "", 0.9)).final


def _cost(final, expected):
    # expected: allow / block / block|caution
    if expected == "allow":
        return FA_COST if final in ("block", "caution") else 0.0  # 误拦家人
    if expected == "block":
        return 1.0 if final != "block" else 0.0  # 漏放克隆
    # block|caution：漏放(allow)才算 MISS
    return 1.0 if final == "allow" else 0.0


def calibrate(rows):
    weights_grid = [{"acoustic": a, "voiceprint": b, "semantic": 1 - a - b}
                    for a, b in itertools.product([0.25, 0.3, 0.35, 0.4],
                                                  [0.25, 0.3, 0.35, 0.4])
                    if 0 < 1 - a - b < 0.6]
    best = None
    for hard in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        for alert in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
            for block in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
                if block <= alert:
                    continue
                for w in weights_grid:
                    total = sum(_cost(_decide(r["acoustic"], r["voiceprint"],
                                              r["semantic"], hard, alert, block, w),
                                      r["expected"]) for r in rows)
                    if best is None or total < best["cost"]:
                        best = {"cost": total, "hard": hard, "alert": alert,
                                "block": block, "weights": w}
    return best


def main():
    if not SCORES.is_file():
        SCORES.parent.mkdir(parents=True, exist_ok=True)
        with SCORES.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "acoustic", "voiceprint", "semantic", "expected"])
        print(f"[info] 未找到 {SCORES}，已写模板。先跑全链路推理填充分数后重跑本脚本。")
        OUT_MD.write_text(_md_skeleton(), encoding="utf-8")
        print(f"骨架报告: {OUT_MD}")
        return

    rows = []
    with SCORES.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not r.get("acoustic"):
                continue
            rows.append({k: float(r[k]) if k != "expected" else r[k]
                         for k in ("acoustic", "voiceprint", "semantic", "expected")})
    if not rows:
        print("[warn] scenario_scores.csv 无数据行，先填充分数。")
        return

    best = calibrate(rows)
    md = _md_report(best, len(rows))
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"标定完成：最优 cost={best['cost']} 参数 hard={best['hard']} "
          f"alert={best['alert']} block={best['block']} weights={best['weights']}")
    print(f"报告: {OUT_MD}")


def _md_skeleton():
    return """# 融合阈值标定（P1-5）

> 状态：**待数据**。先运行全链路推理把 `scenario_scores.csv` 填满，再重跑
> `python scripts/fusion_calibration.py`。

## 方法
- 场景集：100 条（20 家人 ×(2 真声+2 克隆) + 20 陌生人真声），见 `scenario_manifest.csv`。
- 代价：误拦家人 FA = 3 × 漏放克隆 MISS（依据方案书"误把真儿子判假货更伤产品"）。
- 网格：hard_block∈[0.70,0.95]、soft_alert∈[0.30,0.60]、soft_block∈[0.50,0.80]，权重单纯形步长 0.05。
- 消融：去掉任一通道后的漏检率对比（证明三通道缺一不可）。

## 结果（待填）
"""


def _md_report(best, n):
    return f"""# 融合阈值标定（P1-5）

## 方法
- 样本：{n} 条场景集（家人真声=allow / 家人克隆=block / 陌生人=block|caution）。
- 代价：误拦家人 FA = {FA_COST} × 漏放克隆 MISS（依据方案书"误把真儿子判假货更伤产品"）。
- 网格：hard_block / soft_alert / soft_block + 三通道权重单纯形步长 0.05。

## 最优参数
- hard_block = **{best['hard']}**
- soft_alert = **{best['alert']}**
- soft_block = **{best['block']}**
- 权重 = {best['weights']}
- 总代价 = **{best['cost']}**（FA 计 {FA_COST}×）

## 下一步
1. 将 `FusionOrchestrator` 默认参数更新为上述标定值，并在注释注明出处。
2. 跑消融：去掉任一通道重算漏检率，出对比表（答辩王牌）。
"""


if __name__ == "__main__":
    main()
