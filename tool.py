import logging
import json
from datetime import datetime

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # extra로 넘어온 값들 (있을 수도 있고 없을 수도 있음)
        for field in [
            "error_code",
            "service",
            "env",
            "request_id",
            "user_id",
            "path",
            "method",
        ]:
            if hasattr(record, field):
                log[field] = getattr(record, field)

        return json.dumps(log, ensure_ascii=False)


def get_logger():
    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)

    # ⭐ 중복 핸들러 방지 (중요)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

    return logger


# 공통으로 import 해서 쓸 객체
logger = get_logger()