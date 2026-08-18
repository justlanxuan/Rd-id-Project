# E3 Test Contract

```bash
/home/fzliang/miniconda3/envs/reid_project/bin/python \
  experiments/G9:source_to_custom_gap/E3:source_target_matrix/scripts/D1_build_source_target_matrix.py
```

Expected artifact: `/data/fzliang/reid-project/g9/e3_source_target/source_target_matrix.json`.

The index must contain the existing G6 protocol hash, run-record provenance, raw `correct/total`, target-session feature joins, an explicit availability/gate status for every candidate skeleton source, and a list of cells not yet benchmarked.
