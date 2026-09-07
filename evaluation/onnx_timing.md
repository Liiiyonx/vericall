# AASIST ONNX 导出前后推理耗时对比（5.2 验收，2026-09-07）

> 测量方法：同一 3s(48000@16k) 真实音频窗口 ×20 次 warm 推理取 median/mean；GPU 为 RTX5060 8GB。
> 复现：`scripts/eval/export_aasist_onnx.py`（导出+一致性），耗时表命令见日志 / 本文件来源标注。

| 运行时 | 设备 | median | mean | 相对 | 说明 |
|---|---|---|---|---|---|
| torch | CUDA (RTX5060) | **9.6ms** | 9.7ms | 1.0x | 生产 GPU 路径基线 |
| onnxruntime | CPU | **74.7ms** | 74.1ms | ~7.8x | 无 GPU 端侧/一体机可用 |

## 结论
1. **精度无损**：onnxruntime logits 与 torch 最大差 2e-6（softmax 后同）；
2. **CPU 可行**：75ms/窗仍仅为 2s 流式预算的 ~4%，**无 GPU 家庭一体机可端侧跑**（架构卖点：离线/低配终端）；
3. GPU 直跑仍是单窗 9.6ms，瓶颈不在单模型推理而在调度/特征/IO（5.1 并行调度已接 VERICALL_PARALLEL=1 可选）。
4. CAMPPlus（funasr 封装）ONNX 导出算子复杂，待专项；AASIST 已闭环。

## 追加：CAMPPlus ONNX 可行性结论（2026-09-07）
funasr 1.4.14 `AutoModel.export` 对 cam++（rawTorch 类）**不支持**（AttributeError: CAMPPlus has no export）；
需按 cam++ 架构自建计算图并外接 logmel 前端才能导出——成本高、收益小（torch-GPU 路径已足够快，ONNX 卖点由 AASIST 已承接）。
**5.2 收口**：AASIST ✅（1.6MB，CPU 74.7ms）；CAMPPlus 保持 torch，记录可行性结论。
