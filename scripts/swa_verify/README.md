# SWA 正确性验证脚本

方法与结论见 `docs/sglang_flexkv_radixshmem_swa.md` §4.4。全部对着运行中的 sglang 服务跑，需要文档 §2 的
venv 与 sglang 的 `sglang.test.kl_test_utils`；LongBench-v2 首次下载需 `HTTPS_PROXY`。

环境变量：

| 变量 | 含义 |
|---|---|
| `MODEL` | 服务的 `--model-path`，脚本用它加载 tokenizer 构造 prompt（必填） |
| `SGLANG_BASE_URL` | 服务地址，默认 `http://127.0.0.1:30000` |
| `SGLANG_DEBUG_SWA_DUMP_DIR` | `swa_bytes_compare.py` 必填，须与服务启动时的同名变量一致 |
| `FRESH` | `swa_bytes_compare.py`，默认 1：给 prompt 加 256 随机 token 前缀保证未存过 |

运行日志（`*.log`）不入库。

| 脚本 | 用途 |
|---|---|
| `kl_tf_aligned.py` | **正式方法**：页对齐 teacher-forced 打分，host 恢复 vs device 命中配对 KL，24 条 LongBench，~50 s |
| `eyeball.py` | 3 个长 prompt 冷 / host / device 三次输出并排肉眼检查，~40 s |
| `swa_bytes_compare.py` | **字节级方法**：生成并存储 → device 命中 dump 前 → flush（pool 清零）→ host 恢复 → device 命中 dump 后，逐 rank 逐层比对；判据为 sharded D2H 谓词（恢复页第 i 段 == rank i 的第 i 段），另报告 rank 间是否一致。需服务开 `SGLANG_DEBUG_SWA_DUMP_DIR` + `SGLANG_DEBUG_ZERO_SWA_ON_FLUSH=1`。`FRESH=1`（默认）给 prompt 加 256 随机 token 前缀保证未存过 |

早期版本（不页对齐的 `kl_tf.py`、按生成 ids 比较的 `kl_iso.py`、复现 sglang 原方法的
`kl_flexkv.py`）已删除，它们为什么无效见文档 §4.4.1 / §4.4.2。

## 什么是 teacher-forced 打分

普通生成让模型自己逐词往下写，每一步都在"选词"。DSv4 长文本上两个候选词概率经常几乎相等，
一点数值扰动就选到另一个，之后整段分叉——同一请求连发两次，24 条里 21 条在 128 词内就不同了。
比较两次生成的文本，比的是哪一步翻了硬币，不是 KV 对不对。

teacher-forced 打分不让模型写，而是把一段**事先固定的续写**喂给它，只让它对续写的每个词打分：
"在前文条件下你给这个词多大概率"。一次前向、128 个 logprob、没有选词、没有分叉。同一份输入、
同一份 KV，两次打分应几乎一致，于是可以直接检验 KV 内容。

在 `kl_tf_aligned.py` 里：

```
prompt（截到 256 整数倍，如 1792 token） + 固定续写（128 token）
        ↑ 走缓存：来自 FlexKV host 恢复 / SGLang device 命中      ↑ 重新计算并打分
```

请求参数 `max_new_tokens=0`、`logprob_start_len=len(prompt)`。三次打分只差前缀 KV 的来源：
flush 后（FlexKV 恢复，含 SWA 末页）、紧接着（device 命中）、再一次（device 命中，量噪声）。
比较三组 logprob 的 KL：FlexKV 恢复有错则第一组偏离后两组；无错则三组差异同量级。

**必须页对齐**：续写第 1 个 token 的 128 滑窗正好落在缓存最后一页里，即 FlexKV 恢复的那页 SWA。
若 prompt 不是 256 的整数倍，尾部最多 255 个 token 会重新 prefill，续写的窗口全落在这段新算的
token 里，碰不到恢复页，测试对 SWA 无感。

## 字节级比对为什么不是"逐字节相等"

FlexKV D2H 默认 `FLEXKV_KV_SHARED_ACROSS_RANKS_MODE=sharded`：把每页按字节切成 tp 段，rank i 只拷第 i 段
（假定 MLA 的 KV 在各 rank 逐位相同）。SGLang DSv4 在某些长度（实测 L=1536）下各 TP rank 的 SWA 页并不逐位
相同（43 层里仅第 0 层一致，其余约 25% 字节差 1 个值，是 allreduce 级别的舍入抖动），于是存下的页是四段拼接，
恢复后广播给所有 rank。这不是数据搬错，输出影响在噪声内，但"恢复页 == 某个 rank 的原页"不成立，正确谓词是
"恢复页第 i 段 == rank i 原页第 i 段"，脚本按此判定。

负向对照需要 sglang 侧 `flexkv_hybrid_radix_cache.py` 的三个钩子
（`SGLANG_DEBUG_ZERO_SWA_ON_FLUSH`、`SGLANG_DEBUG_ZERO_SWA_TAIL`、`SGLANG_DEBUG_SWA_DUMP_DIR`），目前在 sglang 工作区未提交。
