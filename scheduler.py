from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from crawling_gmaps import crawl_gmaps_reviews
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def job():
    logging.info("Menjalankan crawling mingguan...")
    added = crawl_gmaps_reviews(limit=30)
    logging.info(f"Selesai. Ditambahkan: {added} komentar.")

if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(job, CronTrigger(day_of_week="mon", hour=8, minute=0))
    scheduler.start()
    logging.info("Scheduler berjalan. Tekan CTRL+C untuk berhenti.")

    try:
        # Keep process alive, cek setiap 10 menit
        while True:
            time.sleep(600)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logging.info("Scheduler dimatikan.")
