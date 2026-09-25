#!/usr/bin/env bash
# XLS-R 两列（M2/M3）接续执行器 v2。
#
# 背景：run_full_all.sh 先跑 AASIST 五列（已完成），再跑 RawNet2 三列（进行中），
#       本脚本只等**RawNet2 三列**就绪，再串行跑最贵的 XLS-R 两列（各约 4.3h）。
#       这样即使 RawNet2 编排进程意外中断，XLS-R 也不会与它抢 8G GPU。
#
# 用法：bash scripts/run_xlsr_full_tail.sh
#      tail -f reports/xlsr_tail.log

set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
PY="D:/VeriCall_data/conda_envs/vericall/python.exe"
REP="$ROOT/evaluation/reports"
LOG="$ROOT/reports/xlsr_tail.log"

mkdir -p "$ROOT/reports"
exec > >(tee -a "$LOG") 2>&1

echo "================================================"
echo "XLS-R 尾段接续 v2 启动 $(date '+%F %T')"
echo "等待 RawNet2 三列（rn2_en0/rn2_cn/rn2_lp）完成后开跑"
echo "================================================"

WAIT="rn2_en0 rn2_cn rn2_lp"

# 最多等 10 小时；每 5 分钟检查一次
for i in $(seq 1 120); do
  ready=1
  missing=""
  for t in $WAIT; do
    f="$REP/cfad_scores_clean_${t}_full.jsonl"
    if [ ! -f "$f" ] || [ "$(wc -l < "$f" 2>/dev/null || echo 0)" -lt 62000 ]; then
      ready=0; missing="$missing $t"
    fi
  done
  if [ "$ready" = "1" ]; then
    echo "[$(date '+%T')] RawNet2 三列已全部就绪"
    break
  fi
  if [ $((i % 6)) -eq 0 ]; then
    echo "[$(date '+%T')] 仍在等待，缺:$missing"
  fi
  sleep 300
done

echo
echo ">>> 开始 XLS-R 两列 $(date '+%T')"
for sc in v4 wide; do
  OUT="$REP/cfad_scores_xlsr_${sc}_clean_full.jsonl"
  if [ -f "$OUT" ] && [ "$(stat -c%s "$OUT" 2>/dev/null || echo 0)" -gt 5000000 ]; then
    echo "[skip] xlsr_$sc 已完成"
    continue
  fi
  echo "----- XLS-R scorer=$sc 开始 $(date '+%T') -----"
  "$PY" evaluation/eval_xlsr_full.py --scorer "$sc" || echo "[FAIL] xlsr_$sc（继续）"
  echo "----- XLS-R scorer=$sc 结束 $(date '+%T') -----"
done

echo
echo "================================================"
echo "十列总汇总 $(date '+%F %T')"
echo "================================================"
for t in en0 cn_ft m1lp lp aasist_lpft rn2_en0 rn2_cn rn2_lp; do
  f="$REP/cfad_scores_clean_${t}_full.jsonl"
  [ -f "$f" ] && printf "  %-14s %6d / 62999\n" "$t" "$(wc -l < "$f")" \
               || printf "  %-14s %s\n" "$t" "未生成"
done
for sc in v4 wide; do
  f="$REP/cfad_scores_xlsr_${sc}_clean_full.jsonl"
  [ -f "$f" ] && printf "  %-14s %6d / 62999\n" "xlsr_$sc" "$(wc -l < "$f")" \
               || printf "  %-14s %s\n" "xlsr_$sc" "未生成"
done
echo
echo ">>> 一致性核验"
"$PY" evaluation/verify_full_vs_subset.py
echo
echo "全部结束 $(date '+%F %T')"
