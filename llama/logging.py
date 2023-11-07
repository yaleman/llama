""" log handler things """

import json
import logging
import sys
from typing import Any, Dict


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
                    res: Dict[str, Any] = json.loads(msg)
                except json.JSONDecodeError as error:
                    print(f"Welp, failed to JSON decode {msg}: {error}")
                    res = {"message": msg}
            elif isinstance(msg, list):
                res = {"messages": msg}
            else:
                res = {"message": msg}

        res["level"] = record.levelname

        return json.dumps(
            res,
            default=str,
        )


def setup_logging() -> logging.Logger:
    """sets up logging"""
    logger = logging.getLogger("llama")
    handler = logging.FileHandler("llama_llm_logs.json")
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)
    return logger
