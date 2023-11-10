""" cli for talking to llama-powered steve """
import json
import logging
from pathlib import Path
from random import randint
import sys
from typing import List, Literal, Optional, TypedDict, Union
from uuid import uuid4
import click
import questionary
from llama.generation import ChatPrediction, Dialog, DialogOrderError, Llama, Message
from llama.generation import PromptTooLongError

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


def save_config(config: Config, filepath: Path) -> None:
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
        if isinstance(message, dict):
            role = message.get("role", role)
            message = message.get("content", "").strip()
        if message.strip() == "":
            message = "<empty message>"
        print(f"{prefix}{message}")

        self.logger.info(
            {
                "message": message,
                "role": role,
                "user_name": self.user_name,
                "session_id": self.session_id,
            }
        )

    def ask_for_input(self) -> Optional[Message]:
        """asks a user for input"""
        user_input = questionary.text("What do you want to say?", multiline=True).ask()

        if user_input is None:
            self.logger.info(
                {
                    "message": "Got strange input, skipping it.",
                }
            )
            return None
        self.show_message(user_input, role="user")
        user_message: Message = {
            "role": "user",
            "content": user_input.strip(),
            "dialog_id": self.session_id,
        }
        self.message_history.append(user_message)
        return user_message

    def get_response(self) -> Optional[Message]:
        """get the chat response from the model"""
        print("Generating response...")
        try:
            response: List[ChatPrediction] = self.llama.chat_completion(
                [self.message_history],
                execution_id=self.session_id,
                temperature=self.config["temperature"],
            )
        except DialogOrderError as error:
            self.logger.error(
                {
                    "error": error,
                    "error_type": "DialogOrderError",
                    "session_id": self.session_id,
                }
            )
            return None

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
            return None
        else:
            response_message = this_response["generation"]
            self.message_history.append(response_message)
            self.show_message(response_message)
            return response_message


def question_loop(config: Config, logger: logging.Logger) -> None:
    """do the question loop thing"""
    logger.info({"message": f"Starting up, using model {config['model_dir']}"})

    questionary.print(
        "Hi, we're going to ask for your name, then we're going to start up the model, then start chatting.",
        style="bold fg:yellow",
    )

    user_name = questionary.text("What's your name?").ask()
    steve = Steve(user_name, config, logger)

    while True:
        if steve.ask_for_input() is not None:
            try:
                response = steve.get_response()
                if response is None:
                    break
            except PromptTooLongError as error:
                logger.error(
                    {
                        "message": "Prompt too long, can't deal with this!",
                        "error": str(error),
                        "session_id": steve.session_id,
                    }
                )
                print(error)
                break


@click.command()
@click.option("-c", "--config", "config_filepath", type=click.Path(exists=True))
def main(config_filepath: str = "./llama_steve_config.json") -> None:
    """CLI for talking to Steve"""
    if config_filepath is None:
        config_filepath = "./llama_steve_config.json"
    config = load_config(Path(config_filepath))
    logger = setup_logging()
    question_loop(config, logger)


if __name__ == "__main__":
    main()
