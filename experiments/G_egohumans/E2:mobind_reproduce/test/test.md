# 🧪 E2:mobind_reproduce 测试说明

## 1. 测试目标
确保 MoBInd 官方 EgoHumans 预训练权重在本机环境可加载，官方 cache 可构建，官方 eval 脚本可运行并输出指标。

## 2. 测试对象

| 子实验 | 对象 | 说明 |
|--------|------|------|
| A1 | `mobind_repro` conda env | PyTorch + MoBind 依赖 |
| A2 | `/home/fzliang/MoBind/configs/config.py` 中的 `DATA_ROOT` | 指向 `/data/lyxie/ReID/Data/egohumans` |
| A3 | `preprocess/EgoHumans/cache.py` | contrastive cache |
| A3 | `preprocess/EgoHumans/cache_multi_person.py` | multi-person cache |
| A3 | `preprocess/EgoHumans/cache_sync.py` | sync cache（使用官方 annotations） |
| A4 | `eval_retrieval.py` | 跨模态检索 |
| A4 | `eval_localization.py --task all` | 人员与肢体定位 |
| A4 | `eval_sync_egoh.py --task person/video` | 单人与视频级时序同步 |

## 3. 性能指标与通过阈值

| 测试项 | 通过标准 | 实际结果 | 状态 |
|--------|---------|----------|------|
| 环境安装 | `python -c "import torch; print(torch.cuda.is_available())"` 返回 `True` | True | ✅ |
| 路径对齐 | `os.path.join(DATA_ROOT, 'EgoHumans', 'extracted_data')` 存在可读取的 `.npy` | 456 个 `.npy` | ✅ |
| cache.py | 成功生成 `cache_action_5_2/` 且 train/val/test 非空 | 4659 个样本 | ✅ |
| cache_multi_person.py | 成功生成 `cache_action_multi_5_2/` 且 test 非空 | 299 个样本 | ✅ |
| cache_sync.py | 成功生成 `cache_sync_action_20_5/` 且条目数与 annotations.txt 一致 | 1540 个条目 | ✅ |
| eval_retrieval.py | 输出 R@1/3/5/10/25/50，无 NaN/Inf | R@1 ≈ 0.83，见 results.md | ✅ |
| eval_localization.py | 输出 Person Acc 与 Limb Acc，无 KeyError | Person 98.01%，Limb 89.22% | ✅ |
| eval_sync_egoh.py | 输出 MAE 与 Acc@0.1/0.2/0.5，无越界 | 见 results.md | ✅ |

## 4. 复现实验步骤

```bash
# 一键复现（包含环境激活）
bash experiments/G_egohumans/E2:mobind_reproduce/scripts/A1_run_full_repro.sh
```

或手动执行：

```bash
# 1. 激活环境
conda activate mobind_repro

# 2. 构建 cache
cd /home/fzliang/MoBind
python preprocess/EgoHumans/cache.py --window_sec 5 --stride 2
python preprocess/EgoHumans/cache_multi_person.py --window_sec 5 --stride 2
python preprocess/EgoHumans/cache_sync.py \
  --window_sec 20 --stride_sec 5 \
  --anno_file /home/fzliang/MoBind/data/EgoHumans/cache_sync_action_20_5/annotations.txt

# 3. 运行评测
python eval_retrieval.py --exp_dir ./checkpoints/EgoHumans/stage2_repro
python eval_localization.py --exp_path ./checkpoints/EgoHumans/stage2_repro --task all
python eval_sync_egoh.py --exp_dir ./checkpoints/EgoHumans/stage2_repro --task person
python eval_sync_egoh.py --exp_dir ./checkpoints/EgoHumans/stage2_repro --task video
```

## 5. 失败判定标准

* 环境安装后 `torch.cuda.is_available()` 为 False → 阻塞全部 A4
* cache 构建报错 `FileNotFoundError: extracted_data` → 阻塞 A3
* eval 阶段 `RuntimeError` 或 `KeyError` → 阻塞对应子实验
* 输出指标全为 0 或随机 → 检查 cache split 与 config.data.root_dir 是否对齐

## 6. 已修复问题记录

* `cache.py` / `cache_multi_person.py`：`numpy.int64` 无法 JSON 序列化，已转 Python `int`。
* `builder/build_model.py`：移除了未定义模块导入，修正 `ConvFormer` 路径。
* `eval_sync_egoh.py`：支持 multi-person cache 的 `(P, T, ...)` 形状，并修复 `gt_offsets` 重复追加导致的维度不匹配。
