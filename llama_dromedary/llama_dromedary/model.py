# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    RowParallelLinear,
    ColumnParallelLinear,
    ParallelEmbedding,
    VocabParallelEmbedding,
)


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
    qkv_dim: int = 0
    ffn_dim: int = 0
    model_vocab_size: int = 0
    max_shared_seq_len: int = 0
    use_prefix_cache: bool = False
    use_cache: bool = True


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

        if args.qkv_dim == 0:
            self.head_dim = args.dim // args.n_heads
        else:
            self.head_dim = args.qkv_dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.use_cache = args.use_cache
        if self.use_cache:
            self.cache_k = torch.zeros(
                (args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim)
            ).cuda()
            self.cache_v = torch.zeros(
                (args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim)
            ).cuda()

        self.shared_prefix_length = 0
        self.shared_cache_k = None
        self.shared_cache_v = None
        if args.max_shared_seq_len > 0:
            self.shared_cache_k = torch.zeros(
                (1, args.max_shared_seq_len, self.n_local_kv_heads, self.head_dim)
            ).cuda()
            self.shared_cache_v = torch.zeros(
                (1, args.max_shared_seq_len, self.n_local_kv_heads, self.head_dim)
            ).cuda()

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor], cache_shared_prefix: bool = False):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if not cache_shared_prefix:
            if self.use_cache:
                self.cache_k = self.cache_k.to(xq)
                self.cache_v = self.cache_v.to(xq)

                real_start_pos = start_pos - self.shared_prefix_length
                assert real_start_pos >= 0

                self.cache_k[:bsz, real_start_pos : real_start_pos + seqlen] = xk
                self.cache_v[:bsz, real_start_pos : real_start_pos + seqlen] = xv

                if self.shared_prefix_length == 0:
                    keys = self.cache_k[:bsz, : start_pos + seqlen]
                    values = self.cache_v[:bsz, : start_pos + seqlen]

                else:
                    self.shared_cache_k = self.shared_cache_k.to(xq)
                    self.shared_cache_v = self.shared_cache_v.to(xq)
                    keys = torch.cat(
                        [
                            self.shared_cache_k[:, : self.shared_prefix_length].repeat(bsz, 1, 1, 1),
                            self.cache_k[:bsz, : real_start_pos + seqlen],
                        ],
                        dim=1,
                    )
                    values = torch.cat(
                        [
                            self.shared_cache_v[:, : self.shared_prefix_length].repeat(bsz, 1, 1, 1),
                            self.cache_v[:bsz, : real_start_pos + seqlen],
                        ],
                        dim=1,
                    )
            else:
                # only concat shared cache and current batch
                real_start_pos = start_pos - self.shared_prefix_length
                assert real_start_pos == 0
                if self.shared_prefix_length == 0:
                    keys = xk
                    values = xv
                else:
                    keys = torch.cat(
                        [
                            self.shared_cache_k[:, : self.shared_prefix_length].repeat(bsz, 1, 1, 1),
                            xk,
                        ],
                        dim=1,
                    )
                    values = torch.cat(
                        [
                            self.shared_cache_v[:, : self.shared_prefix_length].repeat(bsz, 1, 1, 1),
                            xv,
                        ],
                        dim=1,
                    )
        else:
            assert self.shared_cache_k is not None
            assert start_pos == 0
            self.shared_cache_k = self.shared_cache_k.to(xq)
            self.shared_cache_v = self.shared_cache_v.to(xq)
            self.shared_cache_k[:, :seqlen] = xk
            self.shared_cache_v[:, :seqlen] = xv
            self.shared_prefix_length = seqlen
            keys = xk
            values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            # score shape: (bs, n_local_heads, slen, cache_len + slen)
            # mask shape: (1, 1, slen, slen)

            if mask.shape[-1] == 1 or mask.shape[-1] == scores.shape[-1]:
                scores = scores + mask
            else:
                cache_len = scores.shape[-1] - mask.shape[-1]
                assert start_pos == cache_len, f"{start_pos} != {cache_len}"
                zeros_mask = torch.zeros_like(mask)[:, :, :, :1].repeat(1, 1, 1, cache_len)
                scores = scores + torch.cat([zeros_mask, mask], dim=-1)

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)

        if args.ffn_dim > 0:
            self.feed_forward = FeedForward(
                dim=args.dim, hidden_dim=args.ffn_dim * 3 // 2, multiple_of=args.multiple_of,
            )
        else:
            self.feed_forward = FeedForward(
                dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of,
                ffn_dim_multiplier=args.ffn_dim_multiplier,
            )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor], cache_shared_prefix: bool = False):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, cache_shared_prefix=cache_shared_prefix)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        if params.model_vocab_size != 0:
            self.tok_embeddings = VocabParallelEmbedding(
                params.model_vocab_size, params.dim, init_method=lambda x: x
            )
        else:
            self.tok_embeddings = ParallelEmbedding(
                params.vocab_size, params.dim, init_method=lambda x: x
            )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        if params.model_vocab_size != 0:
            self.output = ColumnParallelLinear(
                params.dim, params.model_vocab_size, bias=False, init_method=lambda x: x
            )
        else:
            self.output = ColumnParallelLinear(
                params.dim, params.vocab_size, bias=False, init_method=lambda x: x
            )

        if params.qkv_dim == 0:
            self.freqs_cis = precompute_freqs_cis(
                self.params.dim // self.params.n_heads, (self.params.max_seq_len + self.params.max_shared_seq_len) * 2
            )
        else:
            self.freqs_cis = precompute_freqs_cis(
                self.params.qkv_dim // self.params.n_heads, (self.params.max_seq_len + self.params.max_shared_seq_len)  * 2
            )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int,
                cache_shared_prefix: bool = False, return_all_logits: bool = False):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1).type_as(h)          # zhiqings: diagonal: start_pos + 1 ==> 1

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask, cache_shared_prefix=cache_shared_prefix)
        h = self.norm(h)

        if return_all_logits:
            output = self.output(h)  # compute all logits
            if output.shape[-1] != self.vocab_size:
                output = output[:, :, :self.vocab_size]
        else:
            output = self.output(h[:, -1, :])  # only compute last logits

            if output.shape[-1] != self.vocab_size:
                output = output[:, :self.vocab_size]

        return output.float()

    def clear_cache(self):
        for layer in self.layers:
            layer.attention.shared_prefix_length = 0
