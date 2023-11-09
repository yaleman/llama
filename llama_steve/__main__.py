""" cli for talking to llama-powered steve """
import json
import logging
from pathlib import Path
from random import randint
import sys
from typing import List, Literal, TypedDict, Union
from uuid import uuid4
import click
import questionary
from llama.generation import ChatPrediction, Dialog, Llama, Message

from llama_steve import setup_logging


class Config(TypedDict):
    """configuration thing"""

    max_batch_size: int
    nproc_per_node: int
    model_dir: str
    temperature: float
    max_seq_len: int
    tokenizer_path: str


def load_config(filepath: Path) -> Config:
    """load from a path"""
    with filepath.open("r", encoding="utf-8") as fh:
        res: Config = json.load(fh)
    return res


def save_config(config: Config, filepath: Path):
    """save to a path"""
    with filepath.open("w", encoding="utf-8") as fh:
        json.dump(config, fh)


ROLES = Literal["user", "assistant", "system", "error"]


class Steve:
    """chatbot to the stars"""

    def __init__(self, user_name: str, config: Config, logger: logging.Logger) -> None:
        self.logger = logger
        self.config = config
        self.user_name = user_name.strip()
        self.message_history: Dialog = []
        self.session_id = uuid4().hex
        self.llama = Llama.build(
            ckpt_dir=config["model_dir"],
            tokenizer_path=config["tokenizer_path"],
            max_seq_len=config["max_seq_len"],
            max_batch_size=config["max_batch_size"],
            logger=logger,
            seed=randint(0, 100000),
            execution_id=self.session_id,
        )

    def show_message(
        self,
        message: Union[str, Message],
        role: ROLES = "assistant",
    ) -> None:
        """shows a message to the user"""

        if role == "assistant":
            prefix = "Steve: "
        elif role == "user":
            prefix = f"{self.user_name}: "
        elif role == "error":
            prefix = "ERROR: "
        else:
            prefix = f"{role}: "
        if isinstance(message, Message):
            role = message.get("role", role)
            message = message.get("content", "").strip()

        print(f"{prefix}{message}")

        self.logger.info(
            {
                "message": message,
                "role": role,
                "user_name": self.user_name,
                "session_id": self.session_id,
            }
        )

    def ask_for_input(self) -> None:
        """asks a user for input"""
        user_input = questionary.text("What do you want to say?", multiline=True).ask()
        self.show_message(user_input, role="user")
        user_message: Message = {
            "role": "user",
            "content": user_input.strip(),
            "dialog_id": self.session_id,
        }
        self.message_history.append(user_message)

    def get_response(self):
        """get the chat response from the model"""
        response: List[ChatPrediction] = self.llama.chat_completion(
            [self.message_history],
            execution_id=self.session_id,
            temperature=self.config["temperature"],
        )

        if len(response) > 1:
            self.show_message(
                "Got more than one response from the model, using the first one",
                role="error",
            )

        this_response = response[0]
        # log the tokens
        self.logger.info(
            {
                "action": "chat_completion",
                "tokens": this_response.get("tokens", "<no tokens logged>"),
            }
        )

        if "generation" not in this_response:
            self.show_message(
                "There was no message in the response, this is a system error and I must shut down now!",
                role="error",
            )
            sys.exit(1)
        else:
            response_message = this_response["generation"]
            self.message_history.append(response_message)
            self.show_message(response_message)


def question_loop(config: Config, logger: logging.Logger) -> None:
    """do the question loop thing"""
    logger.info({"message": f"Uh, hi there, you're using model {config['model_dir']}"})

    user_name = questionary.text("What's your name?").ask()

    steve = Steve(user_name, config, logger)

    # while True:
    steve.ask_for_input()


@click.command()
@click.option("-c", "--config", "config_filepath", type=click.Path(exists=True))
def main(config_filepath: str = "./llama_steve_config.json"):
    """CLI for talking to Steve"""
    if config_filepath is None:
        config_filepath = "./llama_steve_config.json"
    config = load_config(Path(config_filepath))
    logger = setup_logging()
    question_loop(config, logger)


if __name__ == "__main__":
    main()