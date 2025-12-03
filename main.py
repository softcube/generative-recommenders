# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

"""
Main entry point for model training. Please refer to README.md for usage instructions.
"""

import logging
import os

from typing import List, Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide excessive tensorflow debug messages
import sys

try:
    import fbgemm_gpu  # noqa: F401, E402
except (ImportError, OSError):
    logging.warning(
        "fbgemm_gpu could not be loaded; GPU-optimized sparse ops will be disabled."
    )

import gin

import torch
import torch.multiprocessing as mp


def _install_fbgemm_fallbacks() -> None:
    """
    Install minimal Python fallbacks for a subset of fbgemm ops used in this
    project when the compiled fbgemm_gpu library is unavailable (e.g., macOS).
    These implementations are CPU/MPS-friendly but not optimized.
    """
    try:
        ns = torch.ops.fbgemm
    except Exception:
        return

    # asynchronous_complete_cumsum: [N] -> [N + 1] prefix sum with leading zero.
    if not hasattr(ns, "asynchronous_complete_cumsum"):

        def _async_complete_cumsum(values: torch.Tensor) -> torch.Tensor:
            if values.dim() != 1:
                raise ValueError(
                    f"asynchronous_complete_cumsum fallback expects 1D tensor, "
                    f"got shape {tuple(values.shape)}"
                )
            zeros = torch.zeros(
                1, dtype=values.dtype, device=values.device
            )
            return torch.cat([zeros, torch.cumsum(values, dim=0)], dim=0)

        ns.asynchronous_complete_cumsum = _async_complete_cumsum  # type: ignore[attr-defined]

    # jagged_to_padded_dense: (values, [offsets], [max_lengths], padding_value) -> padded
    if not hasattr(ns, "jagged_to_padded_dense"):

        def _jagged_to_padded_dense(
            values: torch.Tensor,
            offsets,
            max_lengths,
            padding_value: float = 0.0,
        ) -> torch.Tensor:
            if not isinstance(offsets, (list, tuple)) or len(offsets) != 1:
                raise ValueError(
                    "jagged_to_padded_dense fallback expects offsets as a single-element list"
                )
            offsets_tensor = offsets[0]
            if offsets_tensor.dim() != 1:
                raise ValueError(
                    f"offsets tensor must be 1D, got shape {tuple(offsets_tensor.shape)}"
                )
            if isinstance(max_lengths, (list, tuple)):
                if len(max_lengths) != 1:
                    raise ValueError(
                        "jagged_to_padded_dense fallback expects max_lengths as a single-element list"
                    )
                max_len = int(max_lengths[0])
            else:
                max_len = int(max_lengths)

            B = offsets_tensor.numel() - 1
            extra_shape = values.shape[1:]
            out = values.new_full(
                (B, max_len) + extra_shape,
                padding_value,
            )
            for b in range(B):
                start = int(offsets_tensor[b].item())
                end = int(offsets_tensor[b + 1].item())
                length = max(0, min(end - start, max_len))
                if length > 0:
                    out[b, :length] = values[start : start + length]
            return out

        ns.jagged_to_padded_dense = _jagged_to_padded_dense  # type: ignore[attr-defined]

    # dense_to_jagged: inverse of jagged_to_padded_dense for the usage patterns
    # in this repo. Returns (values, offsets) like the C++ op.
    if not hasattr(ns, "dense_to_jagged"):

        def _dense_to_jagged(
            dense: torch.Tensor,
            x_offsets,
            total_L: int = None,
        ):
            if isinstance(x_offsets, (list, tuple)):
                offsets_tensor = x_offsets[0]
            else:
                offsets_tensor = x_offsets
            if offsets_tensor.dim() != 1:
                raise ValueError(
                    f"x_offsets tensor must be 1D, got shape {tuple(offsets_tensor.shape)}"
                )

            B = offsets_tensor.numel() - 1
            lengths = offsets_tensor[1:] - offsets_tensor[:-1]
            values_list = []
            for b in range(B):
                length = int(lengths[b].item())
                if length > 0:
                    values_list.append(dense[b, :length])
            if values_list:
                values = torch.cat(values_list, dim=0)
            else:
                values = dense.new_empty((0,) + dense.shape[2:])

            if total_L is not None and values.shape[0] != total_L:
                if values.shape[0] < total_L:
                    pad_shape = (total_L - values.shape[0],) + values.shape[1:]
                    values = torch.cat(
                        [values, dense.new_zeros(pad_shape)], dim=0
                    )
                else:
                    values = values[:total_L]

            return values, offsets_tensor

        ns.dense_to_jagged = _dense_to_jagged  # type: ignore[attr-defined]


_install_fbgemm_fallbacks()

from absl import app, flags
from generative_recommenders.research.trainer.train import train_fn

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def delete_flags(FLAGS, keys_to_delete: List[str]) -> None:  # pyre-ignore [2]
    keys = [key for key in FLAGS._flags()]
    for key in keys:
        if key in keys_to_delete:
            delattr(FLAGS, key)


delete_flags(flags.FLAGS, ["gin_config_file", "master_port"])
flags.DEFINE_string("gin_config_file", None, "Path to the config file.")
flags.DEFINE_integer("master_port", 12355, "Master port.")
FLAGS = flags.FLAGS  # pyre-ignore [5]


def mp_train_fn(
    rank: int,
    world_size: int,
    master_port: int,
    gin_config_file: Optional[str],
) -> None:
    if gin_config_file is not None:
        # Hack as absl doesn't support flag parsing inside multiprocessing.
        logging.info(f"Rank {rank}: loading gin config from {gin_config_file}")
        gin.parse_config_file(gin_config_file)

    train_fn(rank, world_size, master_port)


def _main(argv) -> None:  # pyre-ignore [2]
    has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0

    if has_cuda:
        world_size = torch.cuda.device_count()
        mp.set_start_method("forkserver")
        mp.spawn(
            mp_train_fn,
            args=(world_size, FLAGS.master_port, FLAGS.gin_config_file),
            nprocs=world_size,
            join=True,
        )
    else:
        logging.warning(
            "No CUDA GPUs detected; running in single-process CPU mode "
            "(this will be slow and some features may be unavailable)."
        )
        world_size = 1
        mp_train_fn(
            rank=0,
            world_size=world_size,
            master_port=FLAGS.master_port,
            gin_config_file=FLAGS.gin_config_file,
        )


def main() -> None:
    app.run(_main)


if __name__ == "__main__":
    main()
