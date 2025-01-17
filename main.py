import functools
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append("picoGPT")

from gpt2 import gpt2, softmax
import gpt2_torch
from utils import load_encoder_hparams_and_params

import torch
from torch import Tensor, nn

from transformers.tokenization_utils_base import BatchEncoding
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel


def max_fn(x):
    x_max = np.where(x > 0, x, 0)
    return x_max / np.sum(x_max)


def sample(p):
    return np.random.choice(np.arange(p.shape[-1]), p=p)


def autoregressive_sampling(x, model, N):
    n = len(x)
    T = len(x) + N

    with tqdm(total=N, desc="autoregressive sampling") as pbar:
        while n < T:
            x = np.append(x, sample(model(x)[-1]))
            n += 1
            pbar.update(1)

    return x


def speculative_sampling(x, draft_model, target_model, N, K):
    # NOTE: paper indexes arrays starting from 1, python indexes from 0, so
    # we have to add an extra -1 term when indexing using n, T, or t
    n = len(x)
    T = len(x) + N

    with tqdm(total=N, desc="speculative sampling") as pbar:
        while n < T:
            prev_n = n

            # Step 1: auto-regressive decode K tokens from draft model and get final p
            x_draft = x
            for _ in range(K):
                p = draft_model(x_draft)
                x_draft = np.append(x_draft, sample(p[-1]))

            # Step 2: target model forward passes on x_draft
            q = target_model(x_draft)

            # Step 3: append draft tokens based on rejection criterion and resample
            # a token on rejection
            all_accepted = True
            for _ in range(K):
                i = n - 1
                j = x_draft[i + 1]
                if np.random.random() < min(1, q[i][j] / p[i][j]):  # accepted
                    x = np.append(x, j)
                    n += 1
                else:  # rejected
                    x = np.append(x, sample(max_fn(q[i] - p[i])))  # resample
                    n += 1
                    all_accepted = False
                    break

            # Step 4: if all draft tokens were accepted, sample a final token
            if all_accepted:
                x = np.append(x, sample(q[-1]))
                n += 1

            # just keeping my sanity
            pbar.update(n - prev_n)
            assert n == len(x), f"{n} {len(x)}"

    return x


# With GPT2 PyTorch
# def speculative_sampling_on_multiple_gpus(x: Tensor, draft_model, target_model, N, K, multi_gpu: bool=False):
#     # NOTE: paper indexes arrays starting from 1, python indexes from 0, so
#     # we have to add an extra -1 term when indexing using n, T, or t
#     if multi_gpu:
#         draft_model.to(device="cuda:0", non_blocking=True)
#         target_model.to(device="cuda:1", non_blocking=True)
#     n = len(x)
#     T = len(x) + N

#     with tqdm(total=N, desc="speculative sampling") as pbar:
#         while n < T:
#             prev_n = n

#             # Step 1: auto-regressive decode K tokens from draft model and get final p
#             x_draft = x
#             for _ in range(K):
#                 p = draft_model(x_draft)
#                 x_draft = np.append(x_draft, sample(p[-1]))
#             if multi_gpu:
#                 x_draft.to(device="cpu", non_blocking=True)

#             # Step 2: target model forward passes on x_draft
#             q = target_model(x_draft)
#             if multi_gpu:
#                 q.to(device="cpu", non_blocking=True)

#             # Step 3: append draft tokens based on rejection criterion and resample
#             # a token on rejection
#             all_accepted = True
#             for _ in range(K):
#                 i = n - 1
#                 j = x_draft[i + 1]
#                 if np.random.random() < min(1, q[i][j] / p[i][j]):  # accepted
#                     x = np.append(x, j)
#                     n += 1
#                 else:  # rejected
#                     x = np.append(x, sample(max_fn(q[i] - p[i])))  # resample
#                     n += 1
#                     all_accepted = False
#                     break

#             # Step 4: if all draft tokens were accepted, sample a final token
#             if all_accepted:
#                 x = np.append(x, sample(q[-1]))
#                 n += 1

#             # just keeping my sanity
#             pbar.update(n - prev_n)
#             assert n == len(x), f"{n} {len(x)}"

#     return x


def speculative_sampling_on_multiple_gpus(
    target_model,
    draft_model,
    target_inputs: BatchEncoding,
    draft_inputs: BatchEncoding,
    N: int,
    K: int,
):
    # NOTE: paper indexes arrays starting from 1, python indexes from 0, so
    # we have to add an extra -1 term when indexing using n, T, or t
    n = len(x)
    T = len(x) + N

    with tqdm(total=N, desc="speculative sampling") as pbar:
        while n < T:
            prev_n = n

            
            # Step 1: auto-regressive decode K tokens from draft model and get final p
            # x_draft = x
            # for _ in range(K):
            #     p = draft_model(x_draft)
            #     x_draft = np.append(x_draft, sample(p[-1]))
            # if multi_gpu:
            #     x_draft.to(device="cpu", non_blocking=True)
            # Tensor.shape = (n_seq + K,)
            x_draft: Tensor = draft_model.generate(**draft_inputs, max_new_tokens=K, do_sample=False).sequences.squeeze().to(device="cuda:1", non_blocking=True)

            # Step 2: target model forward passes on x_draft
            # q = target_model(x_draft)
            # if multi_gpu:
            #     q.to(device="cpu", non_blocking=True)


            # Step 3: append draft tokens based on rejection criterion and resample
            # a token on rejection
            all_accepted = True
            for _ in range(K):
                i = n - 1
                j = x_draft[i + 1]
                if np.random.random() < min(1, q[i][j] / p[i][j]):  # accepted
                    x = np.append(x, j)
                    n += 1
                else:  # rejected
                    x = np.append(x, sample(max_fn(q[i] - p[i])))  # resample
                    n += 1
                    all_accepted = False
                    break

            # Step 4: if all draft tokens were accepted, sample a final token
            if all_accepted:
                x = np.append(x, sample(q[-1]))
                n += 1

            # just keeping my sanity
            pbar.update(n - prev_n)
            assert n == len(x), f"{n} {len(x)}"

    return x


# def create_model_fn(params, hparams, temperature, eps=1e-10):
def create_model_fn(params, hparams, temperature, eps=1e-10, multi_gpu: bool = False):
    model = gpt2_torch.gpt2 if multi_gpu else gpt2
    # f = functools.partial(gpt2, **params, n_head=hparams["n_head"])
    f = functools.partial(model, **params, n_head=hparams["n_head"])

    def model_fn(inputs):
        logits = f(inputs)
        logits = logits / (temperature + eps)  # eps to avoid division by zero
        probs = softmax(logits)
        return probs

    return model_fn


def main(
    prompt: str = "Alan Turing theorized that computers would one day become",
    n_tokens_to_generate: int = 40,
    draft_model_size: str = "124M",
    target_model_size: str = "1558M",
    models_dir: str = "models",
    K: int = 4,
    temperature: float = 0.0,
    seed: int = 123,
    multi_gpu: bool = False,
):
    # seed numpy rng
    np.random.seed(seed)

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, draft_hparams, draft_params = load_encoder_hparams_and_params(
        draft_model_size, models_dir
    )
    _, target_hparams, target_params = load_encoder_hparams_and_params(
        target_model_size, models_dir
    )
    draft_model = create_model_fn(draft_params, draft_hparams)
    target_model = create_model_fn(target_params, target_hparams)

    # encode inputs
    input_ids: list = encoder.encode(prompt)

    def run_sampling_fn(decode_fn, input_ids, **kwargs):
        start = time.perf_counter()
        output_ids = decode_fn(x=input_ids, **kwargs)
        text = encoder.decode(output_ids)
        elapsed_time = time.perf_counter() - start
        return text, elapsed_time

    # autoregressive
    autoregressive_text, autoregressive_time = run_sampling_fn(
        autoregressive_sampling,
        input_ids,
        model=target_model,
        N=n_tokens_to_generate,
    )

    # speculative
    speculative_text, speculative_time = run_sampling_fn(
        speculative_sampling,
        input_ids,
        target_model=target_model,
        draft_model=draft_model,
        N=n_tokens_to_generate,
        K=K,
    )

    if multi_gpu:
        target_model_id: str = "gpt2"
        draft_model_id: str = "distilgpt2"
        
        target_tokenizer = AutoTokenizer.from_pretrained(target_model_id)
        draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_id)
        
        target_inputs = target_tokenizer("Hello, my dog is cute", return_tensors="pt").to(device="cuda:1", non_blocking=True)
        draft_inputs = draft_tokenizer("Hello, my dog is cute", return_tensors="pt").to(device="cuda:0", non_blocking=True)
        
        assert torch.all(draft_inputs.input_ids.cpu() == target_inputs.input_ids.cpu())
        assert torch.all(draft_inputs.attention_mask.cpu() == target_inputs.attention_mask.cpu())

        target_model = GPT2LMHeadModel.from_pretrained(target_model_id).to(device="cuda:1", non_blocking=True)
        # # Set pad_token_id to eos_token_id because GPT2 does not have a PAD token, following
        # # https://github.com/huggingface/transformers/blob/722e9364916e527e8d46cbd828a1516bf6aaebd6/src/transformers/generation/utils.py#L4341C10-L4341C10
        # if target_model.config.pad_token_id is None:
        #     target_model.config.pad_token_id = target_model.config.eos_token_id
        draft_model = GPT2LMHeadModel.from_pretrained(draft_model_id).to(device="cuda:0", non_blocking=True)

        start = time.perf_counter()
        outputs = speculative_sampling_on_multiple_gpus(
            target_model=target_model,
            draft_model=draft_model,
            target_inputs=target_inputs,
            draft_inputs=draft_inputs,
            N=n_tokens_to_generate,
            K=K,
        )
        elapsed_time = time.perf_counter() - start
        text: str = target_tokenizer.batch_decode(outputs)

    # print results
    print()
    print("Autoregressive Decode")
    print("---------------------")
    print(f"Time = {autoregressive_time:.2f}s")
    print(f"Text = {autoregressive_text}")
    print()
    print("Speculative Decode")
    print("------------------")
    print(f"Time = {speculative_time:.2f}s")
    print(f"Text = {speculative_text}")
    if multi_gpu:
        print("Speculative Decode on Multiple GPUs")
        print("------------------")
        print(f"Time = {elapsed_time:.2f}s")
        print(f"Text = {text}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
