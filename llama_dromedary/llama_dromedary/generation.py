# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

# Frequency and presence penalties
# The frequency and presence penalties found in the Completions API can be used to reduce the likelihood of sampling repetitive sequences of tokens. They work by directly modifying the logits (un-normalized log-probabilities) with an additive contribution.
# mu[j] -> mu[j] - c[j] * alpha_frequency - float(c[j] > 0) * alpha_presence
# Where:
# mu[j] is the logits of the j-th token
# c[j] is how often that token was sampled prior to the current position


from typing import List, Dict, Optional, Tuple
import torch
import queue

from llama_dromedary.tokenizer import Tokenizer
from llama_dromedary.model import Transformer
from collections import defaultdict


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.starting_pieces = None

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_gen_len: int = 64,
        logit_bias: Optional[Dict[int, float]] = None,
        echo: bool = False,
        stop: Optional[str] = None,
        unitoken_frequency_penalty: float = 0.0,
        bitoken_frequency_penalty: float = 0.0,
        tritoken_frequency_penalty: float = 0.0,
        quadtoken_frequency_penalty: float = 0.0,
        stream_queue: Optional[queue.Queue] = None,
        frequency_penalty_starts_only: bool = True,
        frequency_penalty_min_range: int = 1024,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        assert max_gen_len <= params.max_seq_len, (max_gen_len, params.max_seq_len)

        stop_tokens_v1 = None
        stop_tokens_v2 = None
        if stop is not None:
            assert bsz == 1, "stop is only supported for single prompt generation"
            stop_tokens_v1 = self.tokenizer.encode(stop, bos=False, eos=False)
            stop_tokens_v1 = torch.tensor(stop_tokens_v1).long().cuda()
            stop_tokens_v2 = self.tokenizer.encode("\n" + stop, bos=False, eos=False)[2:]
            stop_tokens_v2 = torch.tensor(stop_tokens_v2).long().cuda()

        if stream_queue is not None:
            assert bsz == 1, "stream is only supported for single prompt generation"

        if frequency_penalty_starts_only and self.starting_pieces is None:
            self.starting_pieces = self.get_frequency_penalty_set()

        token_seq_freq = [defaultdict(int) for _ in range(bsz)]

        prompt_tokens = []
        for x in prompts:
            # Leave at least $min_gen_len tokens for generation
            max_possible_prompt_len = max(
                params.max_seq_len + params.max_shared_seq_len - max_gen_len,
                params.max_seq_len + params.max_shared_seq_len - min_gen_len)
            while True:
                t = self.tokenizer.encode(x, bos=True, eos=False)
                if len(t) <= max_possible_prompt_len:
                    break

                if params.use_prefix_cache:
                    # if use_prefix_cache, we truncate the last paragrpah of x
                    x = x[: x.rfind("\n")]
                else:
                    # if not, We truncate the first paragrpah of x
                    x = x[x.find("\n") + 1 :]
            prompt_tokens.append(t)

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        # The total length of the input sequence
        total_len = min(params.max_seq_len + params.max_shared_seq_len, max_gen_len + max_prompt_size)
        tokens = torch.full((bsz, total_len), self.tokenizer.eos_id).cuda().long()

        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.eos_id
        start_pos = min_prompt_size
        prev_pos = 0

        if params.use_prefix_cache:
            # find shared prefix tokens in prompts
            for cur_pos in range(start_pos - 2):
                if torch.all(tokens[:, cur_pos] == tokens[0, cur_pos]):
                    prev_pos = cur_pos + 1
                else:
                    break

            # The shared prefix tokens should be less than max_shared_seq_len
            prev_pos = min(prev_pos, params.max_shared_seq_len)

            # Because we can only generate max_seq_len tokens after the shared prefix
            total_len = min(total_len, params.max_seq_len + prev_pos)

            tokens = tokens[:, :total_len]
            # cache shared prefix
            if prev_pos > 0:
                self.model.forward(tokens[:1, :prev_pos], 0, cache_shared_prefix=True)

        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

            if logit_bias is not None:
                for bias_token_id, bias in logit_bias.items():
                    assert bias_token_id < logits.shape[-1]

                    # Inplace update to inference tensor outside InferenceMode is not allowed.
                    logits = logits.clone()
                    logits[..., bias_token_id] += bias

            # Apply frequency penalty
            for frequency_penalty, history_length in [
                (unitoken_frequency_penalty, 0),
                (bitoken_frequency_penalty, 1),
                (tritoken_frequency_penalty, 2),
                (quadtoken_frequency_penalty, 3),
            ]:
                if frequency_penalty > 0.0:
                    for j in range(bsz):
                        if cur_pos > history_length + len(prompt_tokens[j]):
                            total_length = history_length + 1
                            history_token_seq = tuple([_ for _ in token_seq_freq[j] if len(_) == total_length])

                            if history_length == 0:
                                history_token_freq = tuple([token_seq_freq[j][_] for _ in history_token_seq])
                                history_token_seq = tuple([_[0] for _ in history_token_seq])
                            else:
                                history_prefix = tuple(tokens[j, cur_pos - history_length: cur_pos].tolist())
                                # history_token_freq only contains the frequency of the strings that match the prefix
                                history_token_freq = tuple([token_seq_freq[j][_] for _ in history_token_seq
                                                            if _[:-1] == history_prefix])
                                history_token_seq = tuple([_[-1] for _ in history_token_seq if _[:-1] == history_prefix])

                            if len(history_token_seq) > 0:
                                history_token_freq = tuple([
                                    freq if (
                                        not frequency_penalty_starts_only or
                                        (token in self.starting_pieces and token > frequency_penalty_min_range)
                                    ) else 0 for freq, token in zip(history_token_freq, history_token_seq)
                                ])
                                history_token_seq = torch.tensor(history_token_seq).long().cuda()
                                history_token_freq = torch.tensor(history_token_freq).long().cuda()
                                logits = logits.clone()
                                logits[j, history_token_seq] -= frequency_penalty * history_token_freq

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

            # Update token_seq_freq
            for frequency_penalty, history_length in [
                (unitoken_frequency_penalty, 0),
                (bitoken_frequency_penalty, 1),
                (tritoken_frequency_penalty, 2),
                (quadtoken_frequency_penalty, 3),
            ]:
                if frequency_penalty > 0.0:
                    for j in range(bsz):
                        if cur_pos > history_length + len(prompt_tokens[j]):
                            history_prefix = tuple(tokens[j, cur_pos - history_length: cur_pos + 1].tolist())
                            token_seq_freq[j][history_prefix] += 1

            if stop is not None:
                assert len(prompt_tokens) == 1 and tokens.shape[0] == 1
                stop_token_len_v1 = stop_tokens_v1.shape[-1]
                if cur_pos > stop_token_len_v1 + len(prompt_tokens[0]):
                    if torch.all(tokens[0, cur_pos - stop_token_len_v1 : cur_pos] == stop_tokens_v1):
                        break
                stop_token_len_v2 = stop_tokens_v2.shape[-1]
                if cur_pos > stop_token_len_v2 + len(prompt_tokens[0]):
                    if torch.all(tokens[0, cur_pos - stop_token_len_v2 : cur_pos] == stop_tokens_v2):
                        break

            if stream_queue is not None:
                assert len(prompt_tokens) == 1 and tokens.shape[0] == 1
                if cur_pos > len(prompt_tokens[0]):
                    stream_tokens = tokens[0, len(prompt_tokens[0]): cur_pos].tolist()
                    try:
                        stream_tokens = stream_tokens[: stream_tokens.index(self.tokenizer.eos_id)]
                    except ValueError:
                        pass
                    stream_queue.put((stream_tokens, ))

        if params.use_prefix_cache:
            self.model.clear_cache()

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            if echo:
                t = t[: len(prompt_tokens[i]) + max_gen_len]
            else:
                t = t[len(prompt_tokens[i]): len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        if stream_queue is not None:
            stream_queue.put(None)

        return decoded

    def score(
        self,
        prompts: List[str],
        targets: List[str],
        temperature: float = 1.0,
        logit_bias: Optional[Dict[int, float]] = None,
    ) -> List[str]:
        assert len(prompts) == len(targets), "Mismatch between prompts and targets length"

        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(p, bos=True, eos=False) for p in prompts]
        prompted_targets = [self.tokenizer.encode(p + t, bos=True, eos=False) for p, t in zip(prompts, targets)]
        target_tokens = [t[len(p):] for p, t in zip(prompt_tokens, prompted_targets)]

        max_prompt_size = max([len(t) for t in prompt_tokens])
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_target_size = max([len(t) for t in target_tokens])
        total_len = max_prompt_size + max_target_size

        assert total_len <= params.max_seq_len + params.max_shared_seq_len, (
            total_len, params.max_seq_len, params.max_shared_seq_len)
        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()

        for i, (prompt_t, target_t) in enumerate(zip(prompt_tokens, target_tokens)):
            tokens[i, : len(prompt_t)] = torch.tensor(prompt_t).long()
            tokens[i, len(prompt_t): len(prompt_t) + len(target_t)] = torch.tensor(target_t).long()

        shared_prefix_len = 0
        if params.use_prefix_cache:
            # find shared prefix tokens in prompts
            for cur_pos in range(min_prompt_size - 2):
                if torch.all(tokens[:, cur_pos] == tokens[0, cur_pos]):
                    shared_prefix_len = cur_pos + 1
                else:
                    break

            # The shared prefix tokens should be less than max_shared_seq_len
            shared_prefix_len = min(shared_prefix_len, params.max_shared_seq_len)
            # cache shared prefix
            if shared_prefix_len > 0:
                self.model.forward(tokens[:1, :shared_prefix_len], 0, cache_shared_prefix=True)

        assert total_len - shared_prefix_len <= params.max_seq_len, (
            total_len, shared_prefix_len, params.max_seq_len)
        all_logits = self.model.forward(tokens[:, shared_prefix_len:], shared_prefix_len, return_all_logits=True)

        # only compute loss on target tokens
        if logit_bias is not None:
            for bias_token_id, bias in logit_bias.items():
                assert bias_token_id < all_logits.shape[-1]

                # Inplace update to inference tensor outside InferenceMode is not allowed.
                all_logits = all_logits.clone()
                all_logits[..., bias_token_id] += bias

        all_probs = torch.softmax(all_logits / temperature, dim=-1)
        target_log_probs = []

        if params.use_prefix_cache:
            self.model.clear_cache()

        for i, (prompt_t, target_t) in enumerate(zip(prompt_tokens, target_tokens)):
            prompt_len = len(prompt_t)
            target_len = len(target_t)
            target_indices = tokens[i, prompt_len : prompt_len + target_len]

            log_probs_start = prompt_len - 1 - shared_prefix_len
            log_probs_end = prompt_len + target_len - 1 - shared_prefix_len
            log_probs = torch.log(all_probs[i, log_probs_start: log_probs_end, target_indices])
            target_log_probs.append(log_probs.sum().item())
        return target_log_probs

    def get_frequency_penalty_set(self):
        starting_pieces = []
        sp_model = self.tokenizer.sp_model

        for i in range(sp_model.GetPieceSize()):
            if sp_model.IdToPiece(i).startswith("▁") and not sp_model.IdToPiece(i).endswith("▁"):
                starting_pieces.append(i)

        starting_pieces = set(starting_pieces)
        return starting_pieces


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
