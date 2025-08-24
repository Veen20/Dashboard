# crawling_gmaps.py
import logging
import re
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from supabase import create_client
from config import SUPABASE_URL, SUPABASE_KEY

# ====== Konfigurasi ======
URL = "https://www.google.com/maps/place/Samsat+UPTB+Palembang+1/@-2.9870757,104.7412692,17z/data=!4m8!3m7!1s0x2e3b75e6afb58fa1:0xb83c1a47293793d7!8m2!3d-2.9870757!4d104.7438441!9m1!1b1!16s%2Fg%2F11c6rj50mr?entry=ttu&g_ep=EgoyMDI1MDgxOS4wIKXMDSoASAFQAw%3D%3D"

# Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# ===== Fungsi konversi waktu relatif → absolut =====
def convert_relative_time(relative_str: str):
    """
    Mengubah waktu relatif dari Google Maps (dalam bahasa Indonesia)
    menjadi waktu absolut (ISO format).
    """
    if not relative_str:
        return None

    text = relative_str.lower()
    today = datetime.today()

    try:
        if "menit" in text:
            minutes = int(re.search(r"\d+", text).group())
            return (today - timedelta(minutes=minutes)).isoformat()
        elif "jam" in text:
            hours = int(re.search(r"\d+", text).group())
            return (today - timedelta(hours=hours)).isoformat()
        elif "hari" in text:
            days = int(re.search(r"\d+", text).group())
            return (today - timedelta(days=days)).isoformat()
        elif "kemarin" in text:
            return (today - timedelta(days=1)).isoformat()
        elif "minggu" in text:
            weeks = int(re.search(r"\d+", text).group())
            return (today - timedelta(weeks=weeks)).isoformat()
        elif "bulan" in text:
            match = re.search(r"\d+", text)
            months = int(match.group()) if match else 1  # default 1 bulan jika tidak ada angka
            month = today.month - months
            year = today.year
            while month <= 0:
                month += 12
                year -= 1
            day = min(today.day, 28)  # aman untuk Februari
            return today.replace(year=year, month=month, day=day).isoformat()
        elif "setahun" in text:
            day = min(today.day, 28)
            return today.replace(year=today.year - 1, day=day).isoformat()
        elif "tahun" in text:
            years = int(re.search(r"\d+", text).group())
            day = min(today.day, 28)
            return today.replace(year=today.year - years, day=day).isoformat()
        elif "baru saja" in text or "sekarang" in text:
            return today.isoformat()
    except Exception:
        return None

    return None


# ===== Fungsi utama crawling =====
def crawl_gmaps_reviews(limit: int = 20) -> int:
    """
    Crawl review terbaru dari Google Maps dan simpan ke Supabase.
    Parameter:
        limit : jumlah maksimal review yang dicrawl
    Return:
        jumlah review baru yang berhasil ditambahkan
    """
    logging.info(f"Mulai crawling hingga {limit} review...")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(URL)
    wait = WebDriverWait(driver, 10)

    # Klik tombol "Lihat semua ulasan"
    try:
        all_reviews_button = wait.until(
            EC.element_to_be_clickable((By.XPATH, '//button[contains(@aria-label, "ulasan")]'))
        )
        all_reviews_button.click()
        logging.info("Klik tombol 'Lihat semua ulasan' berhasil")
    except Exception as e:
        logging.error(f"Gagal klik tombol ulasan: {e}")
        driver.quit()
        return 0

    # Tunggu div scrollable
    scrollable_div = wait.until(
        EC.presence_of_element_located((By.XPATH, '//div[@role="main"]//div[contains(@class,"m6QErb")]'))
    )

    reviews = []
    while len(reviews) < limit:
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_div)
        WebDriverWait(driver, 2).until(lambda d: True)

        review_elems = driver.find_elements(By.XPATH, '//div[@data-review-id]')
        for r in review_elems:
            try:
                text_elem = r.find_element(By.XPATH, './/span[@class="wiI7pd"]')
                review_text = text_elem.text.strip()
                try:
                    relative_time_elem = r.find_element(By.XPATH, './/span[@class="rsqaWe"]')
                    relative_time = relative_time_elem.text.strip()
                    review_time = convert_relative_time(relative_time)
                except:
                    relative_time, review_time = None, None

                if review_text and review_text not in [rev["comment"] for rev in reviews]:
                    reviews.append({
                        "comment": review_text,
                        "relative_time": relative_time,
                        "review_time": review_time
                    })
                    if len(reviews) >= limit:
                        break
            except:
                continue

        if len(reviews) >= limit:
            break

    driver.quit()
    logging.info(f"Total review diambil: {len(reviews)}")

    # Simpan ke Supabase
    added_count = 0
    for review in reviews:
        exists = supabase.table("comments").select("id").eq("comment", review["comment"]).execute()
        if exists.data:
            continue
        data = {
            "platform": "gmaps",
            "username": "Anonymous",
            "comment": review["comment"],
            "created_at": datetime.now().isoformat(),
            "sentiment": None,
            "relative_time": review["relative_time"],
            "review_time": review["review_time"]
        }
        try:
            supabase.table("comments").insert(data).execute()
            added_count += 1
        except Exception as e:
            logging.error(f"Gagal simpan ke Supabase: {e}")

    logging.info(f"Jumlah review baru yang tersimpan: {added_count}")
    return added_count
