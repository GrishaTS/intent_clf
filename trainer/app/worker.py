import os
import sys
import time
import logging
from datetime import datetime, date
from zoneinfo import ZoneInfo
import requests
from script import retrain
from ctx import Ctx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("trainer.worker")

UTC_0 = ZoneInfo("UTC")

DATA_API_URL = os.getenv("DATA_API_URL").rstrip("/")
BACKEND_API_URL = os.getenv("BACKEND_API_URL")

RUN_EVERY_DAYS = int(os.getenv("RUN_EVERY_DAYS", "14"))
ANCHOR_DATE_STR = os.getenv("ANCHOR_DATE_STR")  # YYYY-MM-DD

def _is_due_today(today: date, anchor: date, period: int) -> bool:
    if period <= 0:
        return True
    return (today - anchor).days >= 0 and ((today - anchor).days % period == 0)

def wait_api_ready(url: str, timeout_s: int = 180) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(url + "/docs", timeout=5)
            if r.status_code < 500:
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"API not ready: {url}")

def _session() -> requests.Session:
    return requests.Session()

# ==================================

def run_once(*, force: bool = False, reason: str = "") -> None:
    now = datetime.now(UTS_0)
    today = now.date()
    anchor = date.fromisoformat(ANCHOR_DATE_STR)

    logger.info("now(UTS_0)=%s force=%s reason=%s", now.isoformat(), force, reason)

    if not force and not _is_due_today(today, anchor, RUN_EVERY_DAYS):
        logger.info("skip: not due (anchor=%s, period=%sd)", anchor, RUN_EVERY_DAYS)
        return

    wait_api_ready(BACKEND_API_URL)

    ctx = Ctx(now=now, data_api_url=DATA_API_URL, session=_session())
    retrain(ctx)
    logger.info("done")

def run_if_due() -> None:
    run_once(force=False, reason="scheduler")

if __name__ == "__main__":
    force = "--force" in sys.argv
    run_once(force=force, reason="cli")
