"""Required-cell matrix for the G6 three-dataset benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

SEEDS = (0, 42, 123)
SOURCES = ("totalcapture", "egohumans")
CUSTOM_FOLDS = {
    1: {
        "train_sessions": ("20260211_172257", "20260211_172522"),
        "val_session": "20260211_171724",
        "test_session": "20260211_171423",
    },
    2: {
        "train_sessions": ("20260211_171423", "20260211_172522"),
        "val_session": "20260211_172257",
        "test_session": "20260211_171724",
    },
    3: {
        "train_sessions": ("20260211_171423", "20260211_171724"),
        "val_session": "20260211_172522",
        "test_session": "20260211_172257",
    },
    4: {
        "train_sessions": ("20260211_171724", "20260211_172257"),
        "val_session": "20260211_171423",
        "test_session": "20260211_172522",
    },
}


@dataclass(frozen=True)
class ExperimentCell:
    job_id: str
    job_type: Literal["train", "evaluate"]
    condition: Literal["source", "zero_shot", "finetune", "direct"]
    dataset: str
    source: str | None
    seed: int
    fold_id: int | None = None
    train_sessions: tuple[str, ...] = ()
    val_session: str | None = None
    test_session: str | None = None
    depends_on: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _source_train_id(source: str, seed: int) -> str:
    return f"train.source.{source}.seed{seed}"


def _target_train_id(condition: str, source: str | None, fold_id: int, seed: int) -> str:
    source_name = source or "none"
    return f"train.{condition}.{source_name}.fold{fold_id}.seed{seed}"


def build_required_cells() -> list[ExperimentCell]:
    cells: list[ExperimentCell] = []
    for source in SOURCES:
        for seed in SEEDS:
            source_train = _source_train_id(source, seed)
            cells.append(
                ExperimentCell(
                    job_id=source_train,
                    job_type="train",
                    condition="source",
                    dataset=source,
                    source=source,
                    seed=seed,
                )
            )
            cells.append(
                ExperimentCell(
                    job_id=f"evaluate.source.{source}.seed{seed}",
                    job_type="evaluate",
                    condition="source",
                    dataset=source,
                    source=source,
                    seed=seed,
                    depends_on=source_train,
                )
            )

            for fold_id, fold in CUSTOM_FOLDS.items():
                test_session = str(fold["test_session"])
                cells.append(
                    ExperimentCell(
                        job_id=(
                            f"evaluate.zero_shot.{source}.fold{fold_id}."
                            f"session{test_session}.seed{seed}"
                        ),
                        job_type="evaluate",
                        condition="zero_shot",
                        dataset="custom",
                        source=source,
                        seed=seed,
                        fold_id=fold_id,
                        test_session=test_session,
                        depends_on=source_train,
                    )
                )

                finetune_train = _target_train_id("finetune", source, fold_id, seed)
                cells.append(
                    ExperimentCell(
                        job_id=finetune_train,
                        job_type="train",
                        condition="finetune",
                        dataset="custom",
                        source=source,
                        seed=seed,
                        fold_id=fold_id,
                        train_sessions=tuple(fold["train_sessions"]),
                        val_session=str(fold["val_session"]),
                        test_session=test_session,
                        depends_on=source_train,
                    )
                )
                cells.append(
                    ExperimentCell(
                        job_id=f"evaluate.finetune.{source}.fold{fold_id}.seed{seed}",
                        job_type="evaluate",
                        condition="finetune",
                        dataset="custom",
                        source=source,
                        seed=seed,
                        fold_id=fold_id,
                        test_session=test_session,
                        depends_on=finetune_train,
                    )
                )

    for seed in SEEDS:
        for fold_id, fold in CUSTOM_FOLDS.items():
            direct_train = _target_train_id("direct", None, fold_id, seed)
            cells.append(
                ExperimentCell(
                    job_id=direct_train,
                    job_type="train",
                    condition="direct",
                    dataset="custom",
                    source=None,
                    seed=seed,
                    fold_id=fold_id,
                    train_sessions=tuple(fold["train_sessions"]),
                    val_session=str(fold["val_session"]),
                    test_session=str(fold["test_session"]),
                )
            )
            cells.append(
                ExperimentCell(
                    job_id=f"evaluate.direct.none.fold{fold_id}.seed{seed}",
                    job_type="evaluate",
                    condition="direct",
                    dataset="custom",
                    source=None,
                    seed=seed,
                    fold_id=fold_id,
                    test_session=str(fold["test_session"]),
                    depends_on=direct_train,
                )
            )
    validate_required_cells(cells)
    return cells


def validate_required_cells(cells: list[ExperimentCell]) -> None:
    ids = [cell.job_id for cell in cells]
    if len(ids) != len(set(ids)):
        raise ValueError("Required-cell matrix contains duplicate job IDs.")
    by_id = {cell.job_id: cell for cell in cells}
    train_cells = [cell for cell in cells if cell.job_type == "train"]
    evaluation_cells = [cell for cell in cells if cell.job_type == "evaluate"]
    if len(train_cells) != 42 or len(evaluation_cells) != 66:
        raise ValueError(
            f"Expected 42 training and 66 evaluation cells, got "
            f"{len(train_cells)} and {len(evaluation_cells)}."
        )
    if {cell.seed for cell in cells} != set(SEEDS):
        raise ValueError(f"Required cells must cover exactly seeds {SEEDS}.")
    for cell in evaluation_cells:
        if not cell.depends_on or cell.depends_on not in by_id:
            raise ValueError(f"Evaluation cell has an unresolved dependency: {cell.job_id}")
        if by_id[cell.depends_on].job_type != "train":
            raise ValueError(f"Evaluation dependency is not a training cell: {cell.job_id}")
    for cell in cells:
        if cell.dataset != "custom":
            continue
        if cell.fold_id not in CUSTOM_FOLDS:
            raise ValueError(f"Custom cell has invalid fold: {cell.job_id}")
        expected_test = CUSTOM_FOLDS[int(cell.fold_id)]["test_session"]
        if cell.test_session != expected_test:
            raise ValueError(f"Custom cell has wrong held-out session: {cell.job_id}")
