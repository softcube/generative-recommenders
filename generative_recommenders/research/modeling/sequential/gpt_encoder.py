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

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from generative_recommenders.research.modeling.sequential.embedding_modules import (
    EmbeddingModule,
)
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
    InputFeaturesPreprocessorModule,
)
from generative_recommenders.research.modeling.sequential.output_postprocessors import (
    OutputPostprocessorModule,
)
from generative_recommenders.research.modeling.sequential.utils import (
    get_current_embeddings,
)
from generative_recommenders.research.modeling.similarity_module import (
    SequentialEncoderWithLearnedSimilarityModule,
)
from generative_recommenders.research.rails.similarities.module import SimilarityModule


class GPTBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        ffn_activation_fn: str,
        attn_dropout_rate: float,
        ffn_dropout_rate: float,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        assert ffn_activation_fn in (
            "relu",
            "gelu",
        ), f"Invalid activation_fn {ffn_activation_fn}"

        self._embedding_dim: int = embedding_dim
        self._num_heads: int = num_heads
        self._ffn_hidden_dim: int = ffn_hidden_dim
        self._ffn_activation_fn: str = ffn_activation_fn
        self._attn_dropout_rate: float = attn_dropout_rate
        self._ffn_dropout_rate: float = ffn_dropout_rate
        self._layer_norm_eps: float = layer_norm_eps

        self._ln1 = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self._attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout_rate,
            batch_first=True,
        )
        self._ln2 = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)

        if ffn_activation_fn == "gelu":
            activation = nn.GELU()
        else:
            activation = nn.ReLU()

        self._ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_hidden_dim),
            activation,
            nn.Dropout(ffn_dropout_rate),
            nn.Linear(ffn_hidden_dim, embedding_dim),
            nn.Dropout(ffn_dropout_rate),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
            attn_mask: [N, N] bool tensor where True entries are masked.
            key_padding_mask: [B, N] bool tensor where True entries are padding.
        Returns:
            [B, N, D]
        """
        # Self-attention block (pre-norm).
        y = self._ln1(x)
        attn_output, _ = self._attn(
            query=y,
            key=y,
            value=y,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_output

        # Feed-forward block (pre-norm).
        z = self._ln2(x)
        ffn_output = self._ffn(z)
        x = x + ffn_output
        return x


class GPTSequentialEncoder(SequentialEncoderWithLearnedSimilarityModule):
    def __init__(
        self,
        max_sequence_len: int,
        max_output_len: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_hidden_dim: int,
        ffn_activation_fn: str,
        attn_dropout_rate: float,
        ffn_dropout_rate: float,
        embedding_module: EmbeddingModule,
        similarity_module: SimilarityModule,
        input_features_preproc_module: InputFeaturesPreprocessorModule,
        output_postprocessor_module: OutputPostprocessorModule,
        activation_checkpoint: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(ndp_module=similarity_module)

        self._embedding_module: EmbeddingModule = embedding_module
        self._embedding_dim: int = embedding_dim
        self._item_embedding_dim: int = embedding_module.item_embedding_dim
        self._max_sequence_length: int = max_sequence_len + max_output_len
        self._input_features_preproc: InputFeaturesPreprocessorModule = (
            input_features_preproc_module
        )
        self._output_postproc: OutputPostprocessorModule = output_postprocessor_module
        self._activation_checkpoint: bool = activation_checkpoint
        self._verbose: bool = verbose

        self._num_layers: int = num_layers
        self._num_heads: int = num_heads
        self._ffn_hidden_dim: int = ffn_hidden_dim
        self._ffn_activation_fn: str = ffn_activation_fn
        self._attn_dropout_rate: float = attn_dropout_rate
        self._ffn_dropout_rate: float = ffn_dropout_rate

        self._blocks: nn.ModuleList = nn.ModuleList(
            [
                GPTBlock(
                    embedding_dim=self._embedding_dim,
                    num_heads=num_heads,
                    ffn_hidden_dim=ffn_hidden_dim,
                    ffn_activation_fn=ffn_activation_fn,
                    attn_dropout_rate=attn_dropout_rate,
                    ffn_dropout_rate=ffn_dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

        # Causal attention mask, sized to max_sequence_len + max_output_len.
        self.register_buffer(
            "_attn_mask",
            torch.triu(
                torch.ones(
                    (self._max_sequence_length, self._max_sequence_length),
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )

        self.reset_state()

    def reset_state(self) -> None:
        for name, params in self.named_parameters():
            if (
                "_input_features_preproc" in name
                or "_embedding_module" in name
                or "_output_postproc" in name
            ):
                if self._verbose:
                    print(f"Skipping initialization for {name}")
                continue
            try:
                nn.init.xavier_normal_(params.data)
                if self._verbose:
                    print(
                        f"Initialize {name} as xavier normal: {params.data.size()} params"
                    )
            except Exception:
                if self._verbose:
                    print(f"Failed to initialize {name}: {params.data.size()} params")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._embedding_module.get_item_embeddings(item_ids)

    def debug_str(self) -> str:
        return (
            f"GPT-d{self._item_embedding_dim}-l{self._num_layers}-h{self._num_heads}"
            + "-"
            + self._input_features_preproc.debug_str()
            + "-"
            + self._output_postproc.debug_str()
            + f"-ffn{self._ffn_hidden_dim}-{self._ffn_activation_fn}"
            + f"-da{self._attn_dropout_rate}-df{self._ffn_dropout_rate}"
            + (f"{'-ac' if self._activation_checkpoint else ''}")
        )

    def _run_one_layer(
        self,
        i: int,
        user_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            i: layer index.
            user_embeddings: [B, N, D]
            valid_mask: [B, N, 1] float (1.0 for valid positions, 0.0 for padding).
        Returns:
            [B, N, D]
        """
        B, N, _ = user_embeddings.size()
        attn_mask = self._attn_mask[:N, :N]
        key_padding_mask = valid_mask.squeeze(-1) == 0
        user_embeddings = self._blocks[i](
            x=user_embeddings,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        user_embeddings = user_embeddings * valid_mask
        return user_embeddings

    def generate_user_embeddings(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            past_ids: [B, N] x int64
        Returns:
            [B, N, D] x float
        """
        past_lengths, user_embeddings, valid_mask = self._input_features_preproc(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )

        for i in range(self._num_layers):
            if self._activation_checkpoint:
                user_embeddings = torch.utils.checkpoint.checkpoint(
                    self._run_one_layer,
                    i,
                    user_embeddings,
                    valid_mask,
                    use_reentrant=False,
                )
            else:
                user_embeddings = self._run_one_layer(i, user_embeddings, valid_mask)

        return self._output_postproc(user_embeddings)

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        batch_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            past_lengths: [B] x int64
            past_ids: [B, N] x int64
            past_embeddings: [B, N, D] x float
            past_payloads: dict of [B, N, ...] tensors
        Returns:
            encoded_embeddings: [B, N, D]
        """
        encoded_embeddings = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )
        return encoded_embeddings

    def encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            past_lengths: [B] x int64
            past_ids: [B, N] x int64
            past_embeddings: [B, N, D] x float
            past_payloads: dict of [B, N, ...] tensors
        Returns:
            [B, D] x float, current sequence embeddings.
        """
        encoded_seq_embeddings = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )
        return get_current_embeddings(
            lengths=past_lengths,
            encoded_embeddings=encoded_seq_embeddings,
        )