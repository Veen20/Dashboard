# z# crawling_gmaps_serpapi.py
# import logging
# import re
# from datetime import datetime, timedelta
# from supabase import create_client
# import streamlit as st
# from serpapi import GoogleSearch

# # ====== Konfigurasi ======
# PLACE_ID = "ChIJoY-1r-Z1Oy4R15M3KUcaPLg"  # ganti dengan Place ID Google Maps Samsat UPTB Palembang 1

# # Supabase client
# SUPABASE_URL = st.secrets["SUPABASE_URL"]
# SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
# SERPAPI_KEY = st.secrets["SERPAPI_KEY"]

# supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# # Logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # ===== Fungsi konversi waktu relatif → absolut (sama seperti kode kamu) =====
# def convert_relative_time(relative_str: str):
#     if not relative_str:
#         return None

#     text = relative_str.lower()
#     today = datetime.today()
#     try:
#         if "menit" in text:
#             minutes = int(re.search(r"\d+", text).group())
#             return (today - timedelta(minutes=minutes)).isoformat()
#         elif "jam" in text:
#             hours = int(re.search(r"\d+", text).group())
#             return (today - timedelta(hours=hours)).isoformat()
#         elif "hari" in text:
#             days = int(re.search(r"\d+", text).group())
#             return (today - timedelta(days=days)).isoformat()
#         elif "kemarin" in text:
#             return (today - timedelta(days=1)).isoformat()
#         elif "minggu" in text:
#             weeks = int(re.search(r"\d+", text).group())
#             return (today - timedelta(weeks=weeks)).isoformat()
#         elif "bulan" in text:
#             match = re.search(r"\d+", text)
#             months = int(match.group()) if match else 1
#             month = today.month - months
#             year = today.year
#             while month <= 0:
#                 month += 12
#                 year -= 1
#             day = min(today.day, 28)
#             return today.replace(year=year, month=month, day=day).isoformat()
#         elif "setahun" in text:
#             day = min(today.day, 28)
#             return today.replace(year=today.year - 1, day=day).isoformat()
#         elif "tahun" in text:
#             years = int(re.search(r"\d+", text).group())
#             day = min(today.day, 28)
#             return today.replace(year=today.year - years, day=day).isoformat()
#         elif "baru saja" in text or "sekarang" in text:
#             return today.isoformat()
#     except Exception:
#         return None

#     return None

# # ===== Fungsi utama crawling =====
# def crawl_gmaps_reviews(limit: int = 150) -> int:
#     logging.info(f"Mulai crawling hingga {limit} review...")

#     params = {
#         "engine": "google_maps_reviews",
#         "type": "place",
#         "place_id": PLACE_ID,
#         "api_key": SERPAPI_KEY
#     }
#     search = GoogleSearch(params)
#     result = search.get_dict()
#     reviews = result.get("reviews", [])[:limit]

#     added_count = 0
#     for r in reviews:
#         comment_text = r.get("text", "")
#         relative_time = r.get("relative_time_description")
#         review_time = convert_relative_time(relative_time)

#         if not comment_text:
#             continue

#         # Cek duplikat
#         # exists = supabase.table("comments").select("id").eq("comment", comment_text).execute()
#         # if exists.data:
#         #     continue

#         data = {
#             "platform": "gmaps",
#             "username": r.get("user", "Anonymous"),
#             "comment": comment_text,
#             "created_at": datetime.now().isoformat(),
#             "sentiment": None,
#             "relative_time": relative_time,
#             "review_time": review_time
#         }
#         try:
#             supabase.table("comments").insert(data).execute()
#             added_count += 1
#         except Exception as e:
#             logging.error(f"Gagal simpan ke Supabase: {e}")

#     logging.info(f"Jumlah review baru yang tersimpan: {added_count}")
#     return added_count


def crawl_gmaps_reviews(limit: int = 150) -> int:
    logging.info(f"Mulai crawling hingga {limit} review...")

    params = {
        "engine": "google_maps_reviews",
        "type": "place",
        "place_id": PLACE_ID,
        "api_key": SERPAPI_KEY
    }
    search = GoogleSearch(params)
    result = search.get_dict()
    reviews = result.get("reviews", [])[:limit]

    added_count = 0
    for r in reviews:
        # Ambil teks review
        comment_text = r.get("text") or r.get("snippet") or ""
        # Ambil nama user
        username = (
            r.get("user", {}).get("name")
            if isinstance(r.get("user"), dict)
            else r.get("user", "Anonymous")
        )
        relative_time = r.get("date") or r.get("relative_time_description")
        review_time = convert_relative_time(relative_time)

        if not comment_text.strip():
            continue

        data = {
            "platform": "gmaps",
            "username": username or "Anonymous",
            "comment": comment_text,
            "created_at": datetime.now().isoformat(),
            "sentiment": None,
            "relative_time": relative_time,
            "review_time": review_time
        }
        try:
            supabase.table("comments").insert(data).execute()
            added_count += 1
        except Exception as e:
            logging.error(f"Gagal simpan ke Supabase: {e}")

    logging.info(f"Jumlah review baru yang tersimpan: {added_count}")
    return added_count
