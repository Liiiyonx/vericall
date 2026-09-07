# 周回归门禁报告（5.5，2026-09-07）

- 执行耗时 4s · 结论 **PASS**

| 项 | 结果 | 备注 |
|---|---|---|
| pytest 全绿 | ✅ | ................                                                         [100%] |
| 红队隔离检查 | ✅ |  |
| 产物 声纹标定 | ✅ | evaluation/voiceprint_calibration.json |
| 产物 语义F1 | ✅ | evaluation/semantic_f1.json |
| 产物 语义校准 | ✅ | evaluation/semantic_calibration.json |
| EER<2% | ✅ | EER=0.005130180593816957 |
| 语义二分类F1>=0.9 | ✅ | F1=0.9869 |
| 红队 meta>=1.6万 | ✅ | rows=16181 |

> GPU/真机项（流式 P90、回放实测）见 streaming/voiceprint 报告，本门禁为纯逻辑+资产门禁。