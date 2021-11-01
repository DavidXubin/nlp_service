import config
from utils import get_celery_topic_app, get_logger

logger = None
if logger is None:
    logger = get_logger("topic-analysis-service")


celery_app = get_celery_topic_app(config)

print("topic analysis initialized")
