"""
chat completion creator
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
"""

import json
import logging
from pathlib import Path
from random import randint
from typing import Any, List, Optional
import uuid

import fire  # type: ignore

from llama.generation import Llama, Dialog
from llama.generation import PromptTooLongError
from llama.logging import setup_logging


def are_dialogs_valid(dialogs: Any, logger: logging.Logger) -> bool:
    """validates the thing works"""
    if len(dialogs) == 0:
        logger.error(
            {"error": "Dialog file must be a list of dialogs, got an empty list"}
        )
        return False
    if not isinstance(dialogs, list):
        logger.error(
            f"Dialog file must be a list of dialogs, got a non-dict: {type(dialogs)}"
        )
        return False
    for dialog in dialogs:
        if not isinstance(dialog, list):
            logger.error(
                {
                    "error": f"Dialog file must be a list of dialogs, got a non-dict: {type(dialog)}"
                }
            )
            return False
        for sub_dia in dialog:
            if not isinstance(sub_dia, dict):
                logger.error(
                    {
                        "error": f"Dialog file must be a list of dialogs, got a non-dict: {type(sub_dia)}"
                    }
                )
                return False
            for key in ["role", "content"]:
                if key not in sub_dia:
                    logger.error({"error": f"Entry missing key: {key}"})
                    return False
    return True


def load_dialogs(
    dialog_filename: Optional[str], logger: logging.Logger
) -> List[Dialog]:
    """load things"""

    dialogs: List[Dialog] = []
    if dialog_filename is None:
        dialog_filepath = Path("default-dialogs.json")
    else:
        dialog_filepath = Path(dialog_filename)

    if not dialog_filepath.exists():
        logger.error({"error": f"Dialog file {dialog_filename} not found"})
        return []

    logger.info({"dialog_filename": dialog_filepath})
    try:
        with dialog_filepath.open("r", encoding="utf-8") as fh:
            dialogs = json.load(fh)

    except json.JSONDecodeError as error:
        logger.error(
            {
                "message": "failed to parse dialog file",
                "filename": dialog_filepath,
                "error": str(error),
            }
        )
        return []
    return dialogs


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    dialog_filename: Optional[str] = "default-dialogs.json",
) -> None:
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts.
            Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences.
            Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    logger = setup_logging()

    execution_id = uuid.uuid4().hex
    random_seed = randint(0, 1000000)

    dialogs = load_dialogs(dialog_filename, logger)
    if not are_dialogs_valid(dialogs, logger):
        return

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        logger=logger,
        seed=random_seed,
        execution_id=execution_id,
    )

    try:
        results = generator.chat_completion(
            dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            execution_id=execution_id,
        )
    except PromptTooLongError as error:
        logger.error(
            {
                "message": "prompt too long",
                "error": str(error),
                "execution_id": execution_id,
            }
        )
        return

    # roll up the input/output things
    for dialog, result in zip(dialogs, results):
        for dialog_idx, msg in enumerate(dialog):
            gen_result = result.get("generation", {})
            logger.info(
                {
                    "action": "chat_content",
                    "dialog_idx": dialog_idx,
                    "request_role": msg.get("role"),
                    "request": msg.get("content"),
                    "dialog_id": msg.get("dialog_id", "<unknown id>"),
                    "response_role": gen_result.get("role", "<unset role>"),
                    "response": gen_result.get("content", "<no content was returned>"),
                    "execution_id": execution_id,
                }
            )


if __name__ == "__main__":
    fire.Fire(main)
