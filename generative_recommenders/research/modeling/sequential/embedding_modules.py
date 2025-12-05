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

import abc

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from generative_recommenders.research.modeling.initialization import truncated_normal


class EmbeddingModule(torch.nn.Module):
    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass


class LocalEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        base_item_embedding_dim: int = 0,
        use_embedding_adapter: bool = False,
        embedding_adapter_hidden_dim: int = 0,
        embedding_adapter_dropout: float = 0.0,
        embedding_adapter_use_bias: bool = True,
        embedding_init_numpy_path: str = "",
    ) -> None:
        super().__init__()

        # Final embedding dimension used by the model.
        self._item_embedding_dim: int = item_embedding_dim
        # Underlying embedding table dimension (can be larger than the final dim).
        self._base_item_embedding_dim: int = (
            base_item_embedding_dim if base_item_embedding_dim > 0 else item_embedding_dim
        )

        self._item_emb = torch.nn.Embedding(
            num_items + 1, self._base_item_embedding_dim, padding_idx=0
        )
        self._adapter: torch.nn.Module
        self._use_adapter: bool = use_embedding_adapter

        if use_embedding_adapter:
            # If no hidden dim specified, default to base embedding dim.
            hidden_dim = (
                embedding_adapter_hidden_dim
                if embedding_adapter_hidden_dim > 0
                else self._base_item_embedding_dim
            )
            layers = []
            input_dim = self._base_item_embedding_dim

            # First adapter layer: base_dim -> hidden_dim.
            layers.append(
                nn.Linear(input_dim, hidden_dim, bias=embedding_adapter_use_bias)
            )
            layers.append(nn.ReLU())
            if embedding_adapter_dropout > 0.0:
                layers.append(nn.Dropout(embedding_adapter_dropout))

            # Second (optional) projection to final embedding dim if needed.
            if hidden_dim != self._item_embedding_dim:
                layers.append(
                    nn.Linear(
                        hidden_dim,
                        self._item_embedding_dim,
                        bias=embedding_adapter_use_bias,
                    )
                )

            self._adapter = nn.Sequential(*layers)
            # L2 normalization layer similar to GPT EmbeddingAdapter.
            self._norm = nn.Identity()
        else:
            self._adapter = nn.Identity()
            self._norm = nn.Identity()

        self.reset_params()

        # Optionally initialize the base embedding matrix from a NumPy file.
        if embedding_init_numpy_path:
            try:
                arr = np.load(embedding_init_numpy_path)
                if arr.ndim != 2:
                    print(
                        f"[WARN] LocalEmbeddingModule: expected 2D numpy array for "
                        f"embedding_init_numpy_path, got shape {arr.shape}; skipping."
                    )
                else:
                    tensor = torch.as_tensor(arr, dtype=self._item_emb.weight.dtype)
                    rows = min(tensor.shape[0], self._item_emb.weight.shape[0])
                    cols = min(tensor.shape[1], self._item_emb.weight.shape[1])
                    if rows <= 0 or cols <= 0:
                        print(
                            f"[WARN] LocalEmbeddingModule: numpy embedding shape "
                            f"{tensor.shape} incompatible with target "
                            f"{self._item_emb.weight.shape}; skipping."
                        )
                    else:
                        self._item_emb.weight.data[:rows, :cols].copy_(
                            tensor[:rows, :cols]
                        )
                        print(
                            f"[INFO] LocalEmbeddingModule: initialized base embeddings "
                            f"from '{embedding_init_numpy_path}' "
                            f"into shape {self._item_emb.weight.shape}"
                        )
            except Exception as e:
                print(
                    f"[WARN] LocalEmbeddingModule: failed to load embeddings from "
                    f"'{embedding_init_numpy_path}': {e}"
                )

        # When using an embedding adapter, freeze the base embedding matrix so
        # only the adapter parameters are updated.
        if self._use_adapter:
            self._item_emb.weight.requires_grad = False

    def debug_str(self) -> str:
        return f"local_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        # Robustness: clamp any out-of-range item ids to the valid range
        # to avoid CUDA device-side asserts when evaluation data contains
        # unseen items. Padding idx is 0, so we map invalid ids there.
        num_embeddings = self._item_emb.num_embeddings
        if torch.any((item_ids < 0) | (item_ids >= num_embeddings)):
            # Log a one-time warning per process.
            if not hasattr(self, "_warned_oob_ids"):
                print(
                    f"[WARN] LocalEmbeddingModule: clamping out-of-range item_ids "
                    f"to [0, {num_embeddings - 1}]. This usually indicates that "
                    f"the eval/test split contains item ids unseen in training."
                )
                self._warned_oob_ids = True  # type: ignore[attr-defined]
            item_ids = item_ids.clamp(min=0, max=num_embeddings - 1)
        embeddings = self._item_emb(item_ids)
        if self._use_adapter:
            # Apply adapter MLP + L2 normalization along the embedding dim.
            orig_shape = embeddings.shape
            embeddings_flat = embeddings.view(-1, self._base_item_embedding_dim)
            adapted = self._adapter(embeddings_flat)
            # L2 normalize
            adapted = F.normalize(adapted, p=2.0, dim=-1, eps=1e-12)
            embeddings = adapted.view(orig_shape)
        return embeddings

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class CategoricalEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        item_id_to_category_id: torch.Tensor,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.register_buffer("_item_id_to_category_id", item_id_to_category_id)
        self.reset_params()

    def debug_str(self) -> str:
        return f"cat_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = self._item_id_to_category_id[(item_ids - 1).clamp(min=0)] + 1
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim
