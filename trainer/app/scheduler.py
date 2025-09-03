from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from zoneinfo import ZoneInfo
import os, sys
import worker

MSK = ZoneInfo("Europe/Moscow")

def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name, str(int(default))).strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def on_start_run_if_needed():
    if _bool_env("RUN_ON_START", False):
        print("[trainer] RUN_ON_START=1 -> run now", flush=True)
        try:
            worker.run_once(force=True, reason="RUN_ON_START")
        except Exception as e:
            print(f"[trainer] RUN_ON_START failed: {e}", flush=True)

def job():
    try:
        worker.run_if_due()
    except Exception as e:
        print(f"[scheduler] job error: {e}", flush=True)

if __name__ == "__main__":
    cron_expr = os.getenv("SCHEDULE_CRON")
    print(f"[scheduler] start (cron={cron_expr})", flush=True)

    on_start_run_if_needed()

    sched = BlockingScheduler(timezone=MSK)
    sched.add_job(job, CronTrigger.from_crontab(cron_expr))
    print(f"[scheduler] scheduled: {cron_expr} (MSK)", flush=True)

    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        print("[scheduler] stop", flush=True)
        sys.exit(0)
