# MLA 当前状态

<!-- mla-loop-next: none -->

**R10 已完成：rebase、整体检视和受影响回归已闭环。** 已将 176 个本地提交回放到最新 `origin/main`（`1486554`），rebase 后基线为 `1ed342c`。原分支 `baafdf5` 保存在 `backup/mla-before-rebase-20261020`，工作区原先干净。

## 本轮改动

普通 FA3 的 metadata 按主仓统一为 2,376 字节、has_softcap 特征及官方参数顺序；删除旧数值兼容层、重复私有 MLA 分派和失效测试。补齐 `math` 引用，移除普通 varlen fake 路径误加的 MLA 校验；Python/C++ 均先分流普通等维、无 qv 输入，修复非分页 TND cache 被 MLA 布局检查错误拒绝的问题，并保留普通 varlen 错误 query 的上游报错优先级。公共测试计时避开 Dynamo 跟踪，清理过时注释、重复默认值/声明及无调用方函数，保留来源和版权。

保留上游普通 FA3 的长度/used-length/512 维修复、动态 HCC 发现和 AICPU 对象隔离；MLA 数值、调度、地址及发射代码保持原基线，CANN/CATLASS 未修改。普通 metadata 签名和 real/fake 契约合入已有 metadata 测试，MLA 仍只有现有公开 E2E 文件。

中英文 README 已移除过期的 MLA 图验收和图测试覆盖说明，与 R8/R9 的普通调用范围一致。

## 环境与构建身份

证据根 `W=/tmp/flash-attention-npu-mla-46928a793876/mla-rebase-review-20261020` 保存三方源码、旧扩展、命令及原始 pytest 日志/JUnit。Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0.beta2；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。普通 FA3 使用 Ascend910B3 物理设备 3，MLA 使用同型号设备 5，均映射逻辑 0。沙箱不能访问 NPU 驱动，实机命令使用 `/tmp/mla-m16-schedule/run.py` 固定环境启动器在主机运行。

最终 v3/910 重建通过（129.15 s），见 `build-final.log`；沿用仓库外 R8 对象目录，重新编译受共享校验头影响的两份 host 单元。实际加载路径为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，两设备 SHA256 均为 `f4f4e1a5d6937cb04250c172825dcc034e88b9889773351eb5bbbaf0e819d8be`，见 `candidate-environment.log`、`mla-environment.log`、`extension-sha256.txt`。156 份生产/构建/测试文件身份记录在 `source-sha256.txt`。

## 当前验证

| 集合 | 本次结果 | 原始证据 |
| --- | --- | --- |
| MLA 代表与成本边界 | 106 passed / 33.39 s | `representative.log/xml` |
| 普通失败复现、输入校验与成本边界 | 113 passed / 46.21 s | `ordinary-final-preflight.log/xml` |
| 完整 MLA 显式参数表 | 1,480 passed / 106.96 s | `mla-acceptance.log/xml` |
| 普通 FA3、metadata、compile、graph 四文件 | 538 passed、13 skipped / 114.49 s | `ordinary-acceptance.log/xml` |

完整集合合计 **2,018 passed、13 skipped，0 failed / errors**。13 项跳过沿用仓库条件：11 项仅适用 Ascend950，1 项 512 维配置不满足 TND query + 分页 KV 条件，1 项 512 维追加 KV 不支持；不计为通过。本次未运行 950/910C 实机或 MLA 图专项。

收集清单为 1,480 MLA + 551 普通 FA3；MLA 节点与 R9 逐项一致，不抽样。普通回归使用仓库 full CI 的 `pytest-xdist -n 2`、8 个 CPU 线程；MLA 使用测试原有单线程 fixture。保留上游独立 reference 和 `assert_fa_close` 阈值；本轮公共 golden 缓存只复用 reference，待测实现每项实际执行。

49 份 MLA 文件 token 复核确认仅注释、两处多余声明、一个无调用方函数及普通输入识别顺序变化；普通前向/Python fake/metadata 后续逻辑与主仓一致，见 `source-audit-final.log`。46 个生成实例（MLA 38）、生成器幂等、AICPU 对象隔离/依赖、CPU 计时夹具、仓库 pre-commit 和 `git diff --check` 通过；三份 MLA 文档及中英文 README 的 379 个相关本地链接/锚点检查通过。

失败和中断如实保留：初轮代表 72 passed、1 failed（`initial-representative.*`）暴露公共计时的 Dynamo 问题；首次普通全量中断于 18 failed、188 passed、9 skipped（`before-tnd-fix-ordinary-acceptance.*`），均为 TND cache 布局误拒绝。仅修正 Python 后仍有 15 项 C++ 前向失败（`python-tnd-fix-ordinary-final-preflight.*`）；同步修复后上述 30 项已全部通过。普通单线程成本预检中断和各旧扩展结果保存在对应日志，不计作最终通过；低精度 CPU reference 标量 GEMM 的耗时通过本轮缓存及仓库双 worker 配方缓解，未修改 reference 或删减用例。

上一版完整普通回归为 7 failed、531 passed、13 skipped / 1796.12 s，失败均是无 qv 的二维 query 被误判为 MLA，覆盖上游 `q must be 3-dimensional` 错误；原始记录在 `before-query-rank-fix-ordinary-acceptance.*`。末段低精度 CPU reference 已完成并保存缓存；错误分派已修正，新扩展的 113 项普通预检包含全部 7 个失败，完整集合亦已通过。对应 106 MLA 代表、30 普通边界和 1,480 MLA 的通过结果一并归档为 `before-query-rank-fix-*`，不计为最终结果。

## 复现与下一步

每项完整命令保存在 `W/*-command.json`，日志封装只执行其中的标准构建/pytest 命令：

```bash
W=/tmp/flash-attention-npu-mla-46928a793876/mla-rebase-review-20261020
python3 "$W/run.py" build-final
python3 "$W/run.py" representative
python3 "$W/run.py" ordinary-final-preflight
python3 "$W/run.py" mla-acceptance
python3 "$W/run.py" ordinary-acceptance
python3 "$W/run.py" quality
```

MLA 完整入口为 `python3 -m pytest -q tests/test_flash_attn_npu_v3_mla.py`；普通完整入口为 `python3 -m pytest -q -n 2 tests/test_flash_attn_npu_v3_metadata.py tests/test_flash_attn_npu_v3.py tests/test_torch_compile_fa3.py tests/test_flash_attn_npu_v3_graph.py`，实际命令另保存 JUnit 与耗时。来源契约及未变覆盖目的继承，旧正确性通过数和性能不累计。本轮未重测固定 CANN 性能，R9 历史闭环由 `git show baafdf5:docs/mla/status.md` 查询。

唯一下一步：**无**（`none`）。已授权范围完成，本轮提交后停止。
