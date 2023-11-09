""" where we hide things """
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from llama.generation import Dialog


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


class JSONFormatter(logging.Formatter):
    """formats things for logging purposes"""

    def format(self, record: logging.LogRecord) -> str:
        """format the message"""
        msg = record.msg

        if isinstance(msg, dict):
            res: Dict[str, Any] = msg
        else:
            if isinstance(msg, str):
                try:
                    res = json.loads(msg)
                except json.JSONDecodeError as error:
                    print(f"Welp, failed to JSON decode {msg}: {error}")
                    res = {"message": msg}
            elif isinstance(msg, list):
                res = {"messages": msg}
            else:
                res = {"message": msg}

        res["level"] = record.levelname
        res["_time"] = datetime.utcnow().isoformat()
        return json.dumps(res, default=str, sort_keys=True, ensure_ascii=False)


def setup_logging() -> logging.Logger:
    """sets up logging"""
    logger = logging.getLogger("llama")
    handler = logging.FileHandler("llama_steve.json")
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    # logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    return logger
