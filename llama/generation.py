"""
Copyright (c) Meta Platforms, Inc. and affiliates.
This software may be used and distributed according
to the terms of the Llama 2 Community License Agreement.

"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict
import uuid

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (  # type: ignore
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer


Role = Literal["system", "user", "assistant"]


class PromptTooLongError(Exception):
    """when your prompt's too long for us to deal with"""


class DialogOrderError(Exception):
    """when your prompt's too long for us to deal with"""


class Message(TypedDict, total=False):
    """message in a dialog"""

    dialog_id: Optional[str]
    role: Role
    content: str


class CompletionPrediction(TypedDict):
    """prediction for a single completion"""

    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    """prediction for a single chat"""

    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required
    completion_id: str


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class Llama:
    """Llama class"""

    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        execution_id: str,
        logger: logging.Logger,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a pre-trained model.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the
            device to CUDA if available, or fails back to gloo if not,
            and loads the pre-trained model and tokenizer.

        """
        logger.info(
            {
                "action": "start",
                "model_dir": ckpt_dir,
                "seed": seed,
                "execution_id": execution_id,
            }
        )

        # no point doing all the startup if we don't have checkpoint files
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint (*.pth) files found in {ckpt_dir}"

        runtime = "gloo"
        try:
            if not torch.distributed.is_initialized():
                # try starting up CUDA first
                torch.distributed.init_process_group("nccl")
                runtime = "nccl"
                torch.cuda.empty_cache()

        except RuntimeError as error:
            logger.debug(
                {
                    "message": "Can't use NCCL (Nvidia CUDA), trying Gloo",
                    "error": error,
                }
            )
            try:
                torch.distributed.init_process_group("gloo")
            except RuntimeError as runtime_error:
                logger.error(
                    {
                        "error": runtime_error,
                        "message": "Can't use Gloo runtime, ran out of runtimes to try, have to quit!",
                    }
                )
                sys.exit(1)

        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if runtime == "nccl":
            torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)
        # if local_rank > 0:
        #     sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"

        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(
            Path(ckpt_dir) / "params.json", "r", encoding="utf-8"
        ) as checkpoint_file_handle:
            params = json.loads(checkpoint_file_handle.read())

        if runtime == "nccl":
            device = "cuda"
        else:
            # https://pytorch.org/docs/stable/tensors.html
            # torch.set_default_tensor_type(torch.DoubleTensor)
            # apple silicon support
            # torch.set_default_dtype(torch.float32)  # type: ignore
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                # don't have to set the default device here because the default is CPU
                # https://pytorch.org/docs/stable/generated/torch.set_default_device.html#torch-set-default-device

                device = "cpu"
                # valid dtypes: https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype
                # torch.set_default_dtype(torch.float64)  # type: ignore
        torch.set_default_device(device)  # type: ignore

        model_args: ModelArgs = ModelArgs(
            runtime=runtime,
            device=device,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(
            model_path=tokenizer_path,
            logger=logger,
            execution_id=execution_id,
        )
        model_args.vocab_size = tokenizer.n_words
        if device == "cuda":
            # pylint: disable=E1101
            torch.set_default_tensor_type(torch.cuda.HalfTensor)  # type: ignore

        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        logger.info(
            {
                "action": "finished loading model",
                "time_s": round(time.time() - start_time, 2),
            }
        )

        return Llama(model, tokenizer, runtime, device, logger)

    def __init__(
        self,
        model: Transformer,
        tokenizer: Tokenizer,
        runtime: str,
        device: str,
        logger: logging.Logger,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.runtime = runtime
        self.device = device
        self.logger = logger

        self.logger.debug(
            {
                "device": self.device,
                "runtime": self.runtime,
            }
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        completion_id: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt
            is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling.
            Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling.
            Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities.
            Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the
            generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]:
                A tuple containing generated token sequences and, if logprobs is True,
                corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text.
            It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """

        self.logger.info(
            {
                "action": "generate_start",
                "completion_id": completion_id,
                "temperature": temperature,
                "top_p": top_p,
                "logprobs": logprobs,
                "max_gen_len": max_gen_len,
                # "prompt_tokens": prompt_tokens, # these are just a bit list of ints, so not much help logging it
            }
        )

        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        if not max_prompt_len <= params.max_seq_len:
            msg = f"Prompt length {max_prompt_len} is longer than max_seq_len parameter {params.max_seq_len}"
            raise PromptTooLongError(msg)
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full(
            (bsz, total_len), pad_id, dtype=torch.long, device=self.device
        )
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.device)

        # store the token probabilities if we're going to log them
        token_logprobs = (
            torch.zeros_like(tokens, dtype=torch.float) if logprobs else None
        )

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=self.device)
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        # iterate over the tokens in the prompt
        for cur_pos in range(min_prompt_len, total_len):
            # send the model to the device to make sure it's up to date
            self.model.to(self.device)

            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                next_token = sample_top_p(
                    torch.softmax(logits[:, -1] / temperature, dim=-1), top_p
                )
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if logprobs and token_logprobs is not None:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs and token_logprobs is not None:
            token_logprobs = token_logprobs.tolist()  # type: ignore
        out_tokens, out_logprobs = [], []

        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs and token_logprobs is not None:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                if logprobs and token_logprobs is not None and probs is not None:
                    probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)

            # this is logging for each dialogue, not per-token
            # self.logger.info(
            #     {
            #         "action": "generate_intermediate",
            #         "step": i,
            #         "tokens": self.tokenizer.decode(out_tokens),
            #     }
            # )
            out_logprobs.append(probs)

        log_result: Dict[str, Any] = {
            "action": "generate_end",
            "completion_id": completion_id,
            # "out_tokens": out_tokens,
        }
        if logprobs:
            log_result["out_logprobs"] = out_logprobs

        self.logger.debug(log_result)

        # typing is hard I'll deal with this later  ... ugh?
        return (out_tokens, out_logprobs if logprobs else None)  # type: ignore

    # pylint: disable=too-many-arguments
    def text_completion(
        self,
        prompts: List[str],
        completion_id: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling.
                Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling.
                Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated
                completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities.
                Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the
                generated output.
                Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the
                generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus
                sampling to introduce
            controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        self.logger.info(
            {
                "action": "text_completion_start",
                "completion_id": completion_id,
                "prompts": prompts,
                "temperature": temperature,
                "top_p": top_p,
                "logprobs": logprobs,
                "max_gen_len": max_gen_len,
            }
        )

        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            completion_id=completion_id,
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            result = [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode([x]) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)  # type: ignore
            ]
        else:
            result = [
                {"generation": self.tokenizer.decode(t)} for t in generation_tokens
            ]

        self.logger.info(
            {
                "action": "text_completion_end",
                "completion_id": completion_id,
                "generation": result,
            },
        )
        return result  # type: ignore

    # pylint: disable=too-many-locals
    def chat_completion(
        self,
        dialogs: List[Dialog],
        execution_id: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        """
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Raises:
            AssertionError: If the last message in a dialog is not from the user.
            AssertionError: If the dialog roles are not in the required 'user', 'assistant', and optional 'system' order.

        Note:
            This method generates assistant responses for the provided conversational dialogs.
            It employs nucleus sampling to introduce controlled randomness in text generation.
            If logprobs is True, token log probabilities are computed for each generated token.

        """

        # this keeps track of a single "completion" requerst

        self.logger.info(
            {
                "action": "completion_start",
                "execution_id": execution_id,
                "temperature": temperature,
                "top_p": top_p,
                "logprobs": logprobs,
                "max_gen_len": max_gen_len,
                "dialogs": len(dialogs),
            }
        )

        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = []
        unsafe_requests = []
        for dialog in dialogs:
            dialog_id = uuid.uuid4().hex
            unsafe_requests.append(
                any(
                    tag in msg.get("content", "")
                    for tag in SPECIAL_TAGS
                    for msg in dialog
                )
            )
            if dialog[0].get("role", "") == "system":
                dialog = [
                    {
                        "dialog_id": dialog_id,
                        "role": dialog[1].get("role"),  # type: ignore
                        # concat the content into a string, wrap the system message in the
                        # system tags, then add the user message onto it
                        "content": "".join(
                            [
                                B_SYS,
                                dialog[0].get("content", ""),
                                E_SYS,
                                dialog[1].get("content", ""),
                            ]
                        ),
                    }
                ]
                # append the rest of the dialog onto the end of it
                for dialog_sub in dialog[2::]:
                    dialog_sub["dialog_id"] = dialog_id
                    dialog.append(dialog_sub)
            # validate the pattern of inputs
            if not all(msg.get("role", "") == "user" for msg in dialog[::2]) and all(
                msg.get("role", "") == "assistant" for msg in dialog[1::2]
            ):
                raise DialogOrderError(
                    "model only supports 'system', 'user' and 'assistant' roles, starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
                )

            for dialog_sub in dialog:
                dialog_sub["dialog_id"] = dialog_id

            self.logger.info(
                {
                    "action": "dialog_input_set",
                    "dialog_id": dialog_id,
                    "execution_id": execution_id,
                    "dialogs": dialog,
                }
            )

            # encode it as a list of ints so we can throw it into the model
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {(prompt.get('content', '')).strip()} {E_INST} {(answer.get('content','')).strip()} ",
                        bos=True,
                        eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )

            assert (
                dialog[-1].get("role", "") == "user"
            ), f"Last message must be from user, got {dialog[-1].get('role', '')}"

            dialog_tokens += self.tokenizer.encode(
                f"{B_INST} {(dialog[-1].get('content', '')).strip()} {E_INST}",
                bos=True,
                eos=False,
            )
            prompt_tokens.append(dialog_tokens)

        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            completion_id=execution_id,
        )

        if logprobs:
            result = [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t)
                        if not unsafe
                        else UNSAFE_ERROR,
                    },
                    "tokens": [self.tokenizer.decode([x]) for x in t],
                    "logprobs": logprobs_i,
                    "execution_id": execution_id,
                }
                for t, logprobs_i, unsafe in zip(
                    generation_tokens, generation_logprobs, unsafe_requests  # type: ignore
                )
            ]
        else:
            result = [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t)
                        if not unsafe
                        else UNSAFE_ERROR,
                        "execution_id": execution_id,
                    }
                }
                for t, unsafe in zip(generation_tokens, unsafe_requests)
            ]

        self.logger.debug(
            {
                "action": "chat_completion_end",
                "completion_id": execution_id,
                "result": result,
            },
        )

        return result  # type: ignore


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
