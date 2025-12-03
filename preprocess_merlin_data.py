# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Simple entry point to preprocess Merlin-style Parquet data
# under tmp/merlin into the SASRec CSV format expected by
# the existing DatasetV2 / RecoDataset pipeline.
#
# Usage:
#   mkdir -p tmp/
#   # Ensure tmp/merlin/train.parquet, valid.parquet, test.parquet exist
#   python3 preprocess_merlin_data.py
#   # Or, to expand each sequence into all prefixes:
#   python3 preprocess_merlin_data.py --expand_sequences_to_prefixes

import argparse

from generative_recommenders.research.data.preprocessor import (
    get_common_preprocessors,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess Merlin Parquet data into SASRec CSV format. "
            "Optionally expand each sequence into all prefix subsequences."
        )
    )
    parser.add_argument(
        "--expand_sequences_to_prefixes",
        action="store_true",
        help=(
            "If set, each original sequence [i1,...,iT] is expanded into "
            "prefixes [i1,i2], [i1,i2,i3], ..., [i1,...,iT] before writing CSVs."
        ),
    )
    args = parser.parse_args()

    dp = get_common_preprocessors()["merlin"]
    # MerlinParquetDataProcessor exposes this flag; default is False.
    if getattr(args, "expand_sequences_to_prefixes", False):
        # pyre-ignore[16]: setting internal configuration flag
        dp._expand_sequences_to_prefixes = True  # type: ignore[attr-defined]
    dp.preprocess_rating()


if __name__ == "__main__":
    main()
