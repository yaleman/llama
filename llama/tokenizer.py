"""tokenizer things """

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import logging
import os
from typing import List

from sentencepiece import SentencePieceProcessor  # type: ignore


class Tokenizer:
    """tokenizing and encoding/decoding text using SentencePiece."""

    def __init__(self, model_path: str, logger: logging.Logger, execution_id: str):
        """
        Initializes the Tokenizer with a SentencePiece model.

        Args:
            model_path (str): The path to the SentencePiece model file.
        """
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.execution_id = execution_id
        self.sp_model = SentencePieceProcessor()
        self.sp_model.LoadFromFile(model_path)
        self.logger = logger
        self.logger.info(
            {
                "action": "reloaded sentencepiece model",
                "filename": model_path,
            }
        )

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        self.logger.info(
            {
                "words": self.n_words,
                "bos_id": self.bos_id,
                "eos_id": self.eos_id,
            }
        )
        assert self.sp_model.vocab_size() == self.sp_model.GetPieceSize()

    def encode(self, input_string: str, bos: bool, eos: bool) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.

        Returns:
            List[int]: A list of token IDs.
        """
        self.logger.info(
            {
                "action": "encode",
                "execution_id": self.execution_id,
                "input": input_string,
            }
        )
        assert isinstance(input_string, str)
        t: List[int] = self.sp_model.Encode(input_string)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """

        # only send it as a list if it's not a single item
        res: str = self.sp_model.Decode(t if len(t) > 1 else t[0])
        self.logger.info(
            {
                "action": "decode",
                "execution_id": self.execution_id,
                "input": t,
                "result": res,
            }
        )
        return res
