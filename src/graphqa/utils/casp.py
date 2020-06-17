from typing import Iterator

import pandas as pd
import numpy as np

HEADER_TEMPLATE = """
PFRMAT QA
TARGET {target_id}
AUTHOR 7411-5180-3518
REMARK GraphQA
METHOD GraphQA
MODEL {stage}
QMODE 2
""".strip()


def scores_df_to_casp(
    stage: int, scores_global: pd.DataFrame, scores_local: pd.DataFrame
) -> Iterator[str]:
    target_ids = scores_global.index.get_level_values("target_id").drop_duplicates()
    for target_id in target_ids:
        target_str = process_target(
            stage=stage,
            target_id=target_id,
            scores_global=scores_global.loc[target_id],
            scores_local=scores_local.loc[target_id],
        )
        yield target_id, target_str


def process_target(
    stage: int, target_id: str, scores_global: pd.DataFrame, scores_local: pd.DataFrame
) -> str:
    header = HEADER_TEMPLATE.format(stage=stage, target_id=target_id)
    decoy_ids = scores_global.index.get_level_values("decoy_id").drop_duplicates()
    body = [
        process_decoy(
            decoy_id=decoy_id,
            scores_global=scores_global.loc[decoy_id],
            scores_local=scores_local.loc[decoy_id],
        )
        for decoy_id in decoy_ids
    ]
    end = "END\n"
    return "\n".join((header, *body, end))


def process_decoy(decoy_id, scores_global, scores_local) -> str:
    gdtts = scores_global["gdtts"]
    lddt = scores_local["lddt"]
    distance = convert_lddt_to_distance(lddt)
    distance_str = ""
    for i, d in enumerate(distance):
        distance_str += f"{d:.3f} "
        if (i + 1) % 20 == 0:
            distance_str += "\n"
    return f"{decoy_id} {gdtts:.3f} {distance_str}"


def convert_lddt_to_distance(lddt: np.ndarray) -> np.ndarray:
    return np.clip(5 * np.sqrt(1 / lddt - 1), a_min=0, a_max=15)
