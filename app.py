"""
====================================================================================================
SISTEM KLASIFIKASI BERITA OTOMATIS (SPORT, FINANCE, OTHER)
MENGGUNAKAN WORD2VEC SKIP-GRAM DAN GAUSSIAN NAIVE BAYES
====================================================================================================
Deskripsi Proyek:
Aplikasi berbasis Flask untuk melakukan klasifikasi kategori berita secara otomatis
langsung dari tautan (URL) artikel web berita berbahasa Indonesia.

Alur Pipeline Utama (End-to-End):
[INPUT URL] 
    
[TAHAP 1: Web Scraping] -> Mengambil judul & isi paragraf artikel dari HTML web.
    
[TAHAP 2: Text Preprocessing] -> Pembersihan teks: Case Folding, Cleansing, Tokenizing, & Stopword Removal (Tanpa Stemming).
    
[TAHAP 3: Vektorisasi Fitur] -> Word2Vec Skip-Gram (Menghitung rata-rata bobot vektor dokumen 100-dimensi & Vocab Coverage).
    
[TAHAP 4: Deteksi Domain & OOD] -> Analisis kata kunci untuk menyaring berita di luar ranah Sport & Finance ('Other').
    
[TAHAP 5: Normalisasi & Klasifikasi] -> MinMaxScaler Scaling + Gaussian Naive Bayes Classifier (Prediksi & Probabilitas).
    
[TAHAP 6: Output / UI Web] -> Menampilkan hasil prediksi, confidence score, token, dan metadata ke halaman web.
====================================================================================================
"""

import os
import re
import pickle
import requests
import numpy as np
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, redirect, url_for, flash
from dotenv import load_dotenv

# Memuat variabel lingkungan dari file .env (misalnya SECRET_KEY, PORT)
load_dotenv()

# Inisialisasi Aplikasi Flask
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "news_classifier_pipeline_secret_key_2026")


# ==================================================================================================
# 1. INISIALISASI & LOAD MODEL MACHINE LEARNING TERLATIH
# ==================================================================================================
# Penjelasan Alur:
# Pada tahap awal saat server dinyalakan, aplikasi memuat artefak model yang sudah disimpan 
# dalam format pickle (.pkl). Artefak ini berisi:
#   1. skipgram_model : Model Word2Vec dengan arsitektur Skip-Gram untuk representasi semantik kata.
#   2. scaler         : MinMaxScaler untuk menormalisasi nilai vektor ke rentang yang sesuai.
#   3. naive_bayes    : Model Gaussian Naive Bayes untuk menghitung probabilitas kelas kategori.
#   4. hyperparameter : vector_size (100 dimensi), window size (5 kata), dan epochs pelatihan.
# ==================================================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_skipgram_naive_bayes (1).pkl")

skipgram_model = None
scaler = None
naive_bayes = None
vector_size = 100
window = 5
epochs = 10
model_error = None

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as file:
            saved_model = pickle.load(file)

        skipgram_model = saved_model.get("skipgram_model")
        scaler = saved_model.get("scaler")
        naive_bayes = saved_model.get("naive_bayes")
        vector_size = saved_model.get("vector_size", 100)
        window = saved_model.get("window", 5)
        epochs = saved_model.get("epochs", 10)
    else:
        model_error = f"File model tidak ditemukan di lokasi: {MODEL_PATH}"
except Exception as e:
    model_error = f"Gagal memuat file model machine learning: {str(e)}"


# ==================================================================================================
# 2. DAFTAR STOPWORDS & KAMUS KATA KUNCI DOMAIN
# ==================================================================================================
# Penjelasan Alur:
# - STOPWORDS: Kumpulan kata fungsional dalam Bahasa Indonesia yang memiliki frekuensi tinggi namun
#   tidak membawa informasi spesifik mengenai topik/kategori berita (misal: "yang", "dan", "di").
# - SPORT_KEYWORDS & FINANCE_KEYWORDS: Digunakan untuk verifikasi domain guna mendeteksi berita
#   yang berada di luar cakupan pelatihan ("Other" / Out-Of-Distribution), sehingga model biner tidak
#   salah memprediksi berita topik umum (seperti kuliner/politik) menjadi sport atau finance.
# ==================================================================================================

STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan",
    "ini", "itu", "pada", "adalah", "akan", "dalam",
    "juga", "karena", "oleh", "sebagai", "atau", "tersebut",
    "bahwa", "saat", "lebih", "sudah", "telah", "bagi",
    "agar", "hingga", "sehingga", "setelah", "sebelum",
    "dapat", "bisa", "tidak", "ada", "mereka", "kami",
    "kita", "anda", "ia", "dia", "nya", "para", "sebuah",
    "seorang", "terhadap", "menjadi", "masih", "hanya",
    "sangat", "jika", "ketika", "namun", "tetapi",
    "secara", "yakni", "bahkan", "salah", "satu"
}

SPORT_KEYWORDS = {
    "olahraga", "sport", "sepak", "bola", "futsal", "basket", "voli",
    "badminton", "bulu", "tangkis", "tenis", "atlet", "atletik", "lari",
    "renang", "tinju", "petinju", "gulat", "balap", "motor", "motogp",
    "formula", "pemain", "pelatih", "tim", "klub", "liga", "turnamen",
    "kompetisi", "pertandingan", "laga", "gol", "menang", "kalah", "juara",
    "skor", "piala", "olimpiade", "medali", "persib", "persija", "persebaya",
    "pssi", "premier", "league", "champions", "uefa", "fifa", "nba", "nfl",
    "mlb", "ufc", "moto"
}

FINANCE_KEYWORDS = {
    "keuangan", "finance", "finansial", "ekonomi", "ekonom", "bisnis",
    "investasi", "investor", "saham", "obligasi", "reksadana", "bank",
    "perbankan", "bursa", "pasar", "modal", "rupiah", "dolar", "usd",
    "valuta", "kurs", "inflasi", "deflasi", "suku", "bunga", "pajak",
    "pendapatan", "laba", "rugi", "profit", "perusahaan", "emiten",
    "dividen", "aset", "utang", "kredit", "pinjaman", "devisa", "ekspor",
    "impor", "produksi", "konsumsi", "moneter", "fiskal", "dagang",
    "perdagangan", "startup", "ipo", "market", "stock", "trading",
    "fund", "cryptocurrency", "kripto"
}

STRONG_SPORT_KEYWORDS = {
    "sepak", "bola", "futsal", "basket", "voli", "badminton", "tangkis",
    "tenis", "atlet", "atletik", "petinju", "tinju", "gulat", "motogp",
    "formula", "pemain", "pelatih", "tim", "klub", "liga", "turnamen",
    "pertandingan", "laga", "gol", "piala", "olimpiade", "medali", "pssi",
    "fifa", "uefa", "nba", "nfl", "mlb", "ufc"
}

STRONG_FINANCE_KEYWORDS = {
    "saham", "obligasi", "reksadana", "investasi", "investor", "bursa",
    "bank", "perbankan", "rupiah", "dolar", "kurs", "inflasi", "deflasi",
    "suku", "bunga", "pajak", "laba", "rugi", "profit", "emiten",
    "dividen", "aset", "utang", "kredit", "pinjaman", "moneter", "fiskal",
    "ipo", "trading", "market", "stock", "cryptocurrency", "kripto"
}


# ==================================================================================================
# 3. PIPELINE TAHAP 1: WEB SCRAPING BERITA DARI URL
# ==================================================================================================
# Penjelasan Alur:
# 1. Menerima URL berita yang diinputkan oleh pengguna pada form.
# 2. Mengirimkan HTTP Request ke server berita dengan custom header User-Agent agar tidak diblokir.
# 3. Parsing kode HTML halaman menggunakan BeautifulSoup.
# 4. Mengekstrak judul artikel (mencari tag <h1> atau <title>).
# 5. Membersihkan tag yang tidak relevan dengan teks berita (script, style, navigasi, footer, dsb).
# 6. Mengambil seluruh teks artikel dari tag paragraf (<p>) dan menggabungkannya menjadi 1 string utuh.
# ==================================================================================================

def scrape_article_from_url(url):
    """
    Mengambil judul dan isi artikel berita dari link URL web.
    Return: tuple (article_title, article_text)
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }
        
        # Mengirim request HTTP GET dengan batas waktu (timeout) 15 detik
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        # Parsing isi halaman web dengan BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Ekstraksi Judul Berita
        title = ""
        title_tag = soup.find("h1") or soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        # Hapus elemen HTML non-artikel (iklan, menu navigasi, javascript, style)
        for junk in soup(["script", "style", "nav", "footer", "header", "aside"]):
            junk.decompose()

        # Ekstraksi semua isi paragraf (<p>) teks artikel
        paragraphs = soup.find_all("p")
        raw_text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
        
        return title, raw_text.strip()
    except Exception as error:
        return "", f"Gagal mengambil artikel dari URL: {str(error)}"


# ==================================================================================================
# 4. PIPELINE TAHAP 2: TEXT PREPROCESSING (TANPA STEMMING)
# ==================================================================================================
# Penjelasan Alur Preprocessing:
# Preprocessing bertujuan mengubah teks mentah hasil scraping menjadi token kata yang bersih 
# dan siap diproses oleh algoritma machine learning.
#
# Tahapan Preprocessing yang diterapkan:
#   Sub-Tahap 2.1 - Case Folding & Cleansing:
#     - Mengubah semua karakter huruf menjadi huruf kecil (lowercase).
#     - Menghilangkan link URL (http/https/www).
#     - Menghilangkan tag HTML (<...>).
#     - Menghilangkan angka (0-9) dan karakter khusus/tanda baca, hanya menyisakan huruf alfabet [a-z].
#     - Menghapus spasi berlebih (whitespace normalization).
#
#   Sub-Tahap 2.2 - Tokenizing:
#     - Memecah rangkaian kalimat menjadi daftar token kata tunggal berdasarkan pemisah spasi.
#
#   Sub-Tahap 2.3 - Stopword Removal:
#     - Memfilter dan membuang kata-kata umum (stopword) yang tidak memberikan kontribusi makna topik.
#
#   *Catatan Khusus Pengujian & Akademik:
#    Proses Stemming DIHAPUS. Model Word2Vec Skip-Gram bekerja optimal menangkap relasi kontekstual
#    kata, dan peniadaan stemming secara signifikan meningkatkan performa komputasi (respon jauh
#    lebih cepat) tanpa mengorbankan akurasi klasifikasi.
# ==================================================================================================

def clean_text(text):
    """
    Sub-Tahap 2.1: Case Folding & Cleansing Teks Mentah.
    Menghilangkan URL, HTML, angka, dan karakter non-alfabet.
    """
    # Mengubah ke huruf kecil (Case Folding)
    text = str(text).lower()
    
    # Menghapus URL dan link web
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    
    # Menghapus tag HTML yang tersisa
    text = re.sub(r"<.*?>", " ", text)
    
    # Menghapus angka
    text = re.sub(r"\d+", " ", text)
    
    # Menghapus tanda baca dan karakter khusus (hanya menyimpan karakter alfabet a-z dan spasi)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    
    # Merapikan spasi ganda menjadi spasi tunggal
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def tokenize(text):
    """
    Sub-Tahap 2.2: Tokenizing.
    Memecah string teks bersih menjadi list kata (tokens).
    """
    return text.split()

def remove_stopwords(tokens):
    """
    Sub-Tahap 2.3: Stopword Removal.
    Menyaring dan menghapus kata-kata umum dari list tokens.
    """
    return [token for token in tokens if token not in STOPWORDS]

def preprocess_pipeline(raw_text):
    """
    Fungsi Penggabung Tahap Preprocessing (End-to-End Preprocessing tanpa Stemming):
    Raw Text -> Case Folding & Cleansing -> Tokenizing -> Stopword Removal -> Clean Tokens
    """
    cleaned_string = clean_text(raw_text)
    tokens = tokenize(cleaned_string)
    filtered_tokens = remove_stopwords(tokens)
    return filtered_tokens


# ==================================================================================================
# 5. PIPELINE TAHAP 3: EKSTRAKSI FITUR DENGAN WORD2VEC SKIP-GRAM
# ==================================================================================================
# Penjelasan Alur Vektorisasi:
# 1. Word2Vec Skip-Gram merepresentasikan setiap kata individual ke dalam vektor kontinu berdimensi 100.
# 2. Vocabulary Coverage:
#    Menghitung seberapa banyak persentase token dari artikel yang terdapat pada kamus model Skip-Gram.
#    Rumus: (Jumlah Kata Dikenali / Total Kata Artikel) * 100%
# 3. Document Vector Averaging (Mean Pooling):
#    Sebuah artikel berita terdiri dari banyak kata. Untuk mendapatkan 1 representasi vektor tunggal 
#    dari keseluruhan dokumen artikel, kita mengambil rata-rata (mean) dari semua vektor kata:
#        V_dokumen = (1 / N) * SUM( V_kata_i )
#    Jika tidak ada kata yang dikenali, dihasilkan vektor nol (zero vector) berukuran 100 dimensi.
# ==================================================================================================

def calculate_vocabulary_coverage(tokens, model):
    """
    Menghitung persentase kata dalam artikel yang dikenali oleh kamus model Word2Vec.
    """
    if len(tokens) == 0 or model is None or not hasattr(model, "wv"):
        return 0.0
    known_tokens = [token for token in tokens if token in model.wv]
    return (len(known_tokens) / len(tokens)) * 100.0

def calculate_document_vector(tokens, model):
    """
    Mengubah list token menjadi representasi 1 vektor dokumen 100 dimensi
    menggunakan metode perataan (mean pooling) vektor kata Word2Vec Skip-Gram.
    """
    vectors = []
    if model is not None and hasattr(model, "wv"):
        for token in tokens:
            if token in model.wv:
                vectors.append(model.wv[token])
    
    # Jika tidak ada kata yang cocok di kamus model, kembalikan vektor 0
    if len(vectors) == 0:
        return np.zeros(vector_size)
    
    # Hitung nilai rata-rata dari seluruh vektor kata
    return np.mean(vectors, axis=0)


# ==================================================================================================
# 6. PIPELINE TAHAP 4: DETEKSI DOMAIN & OUT-OF-DISTRIBUTION (OOD GUARD)
# ==================================================================================================
# Penjelasan Alur:
# Model klasifikasi dilatih secara khusus untuk membedakan kategori 'Sport' dan 'Finance'.
# Apabila pengguna menginputkan artikel tentang kategori lain (misal: Politik, Hiburan, atau Wisata),
# sistem melakukan deteksi berbasis kata kunci topik untuk menandai artikel sebagai 'Other'
# agar sistem tidak memaksakan klasifikasi yang salah (False Positive).
# ==================================================================================================

def detect_article_domain(tokens):
    """
    Menganalisis kemunculan kata kunci domain pada artikel.
    Return: (detected_domain, matches_dictionary)
    """
    token_set = set(tokens)
    
    # Mencari irisan kata antara artikel dengan kamus domain
    sport_matches = token_set.intersection(SPORT_KEYWORDS)
    finance_matches = token_set.intersection(FINANCE_KEYWORDS)
    strong_sport_matches = token_set.intersection(STRONG_SPORT_KEYWORDS)
    strong_finance_matches = token_set.intersection(STRONG_FINANCE_KEYWORDS)

    matches_data = {
        "sport": sorted(list(sport_matches)),
        "finance": sorted(list(finance_matches)),
        "strong_sport": sorted(list(strong_sport_matches)),
        "strong_finance": sorted(list(strong_finance_matches))
    }

    sport_count = len(sport_matches)
    finance_count = len(finance_matches)
    strong_sport_count = len(strong_sport_matches)
    strong_finance_count = len(strong_finance_matches)

    # 1. Jika tidak ditemukan satupun kata kunci domain Sport maupun Finance -> 'Other'
    if sport_count == 0 and finance_count == 0:
        return "other", matches_data

    # 2. Prioritas pertama: kecocokan kata kunci kuat (strong keyword)
    if strong_sport_count >= 1 and strong_sport_count > strong_finance_count:
        return "sport", matches_data

    if strong_finance_count >= 1 and strong_finance_count > strong_sport_count:
        return "finance", matches_data

    # 3. Prioritas kedua: jumlah total kecocokan kata kunci
    if sport_count >= 2 and sport_count > finance_count:
        return "sport", matches_data

    if finance_count >= 2 and finance_count > sport_count:
        return "finance", matches_data

    # Jika skor seimbang atau tidak signifikan, klasifikasikan sebagai 'other'
    return "other", matches_data


# ==================================================================================================
# 7. PIPELINE TAHAP 5: KLASIFIKASI DENGAN GAUSSIAN NAIVE BAYES
# ==================================================================================================
# Penjelasan Alur:
# 1. Fitur Vektor Dokumen (100 dimensi) diskalakan menggunakan MinMaxScaler (scaler.transform).
# 2. Model Gaussian Naive Bayes menghitung probabilitas posterior untuk setiap kelas target:
#        P(Kelas | Dokumen) = P(Kelas) * PROD( P(Fitur_i | Kelas) )
# 3. Menghasilkan label kategori terpilih (Sport / Finance) dan persentase tingkat keyakinan (Confidence Score).
# ==================================================================================================

def predict_article_category(doc_vector, detected_domain):
    """
    Melakukan normalisasi fitur dan prediksi kelas menggunakan Gaussian Naive Bayes.
    Return: tuple (final_label, confidence_score, probabilities_dict, status_note)
    """
    # Skenario 1: Artikel terdeteksi di luar domain (Other)
    if detected_domain == "other":
        final_label = "Other"
        confidence = 100.0
        probabilities = {"Sport": "0.00%", "Finance": "0.00%", "Other": "100.00%"}
        status_note = "Berita ini dikategorikan sebagai 'Other' (di luar topik Sport & Finance) karena tidak ditemukan indikator domain yang mencukupi."
        return final_label, confidence, probabilities, status_note

    # Skenario 2: Prediksi menggunakan Model Gaussian Naive Bayes
    if naive_bayes is not None and scaler is not None and not np.all(doc_vector == 0):
        try:
            # Standarisasi / Skalasi vektor dokumen
            doc_vector_scaled = scaler.transform([doc_vector])
            
            # Prediksi kelas dan distribusi probabilitas
            raw_prediction = naive_bayes.predict(doc_vector_scaled)[0]
            raw_probabilities = naive_bayes.predict_proba(doc_vector_scaled)[0]

            # Format label hasil prediksi
            pred_str = str(raw_prediction).lower()
            if pred_str in ["sport", "0"]:
                final_label = "Sport"
            elif pred_str in ["finance", "1"]:
                final_label = "Finance"
            else:
                final_label = str(raw_prediction).capitalize()

            # Tingkat keyakinan (probabilitas tertinggi dalam persen)
            confidence = float(np.max(raw_probabilities)) * 100.0

            # Susun kamus distribusi probabilitas per kelas
            probabilities = {}
            for cls_name, prob in zip(naive_bayes.classes_, raw_probabilities):
                cls_clean = str(cls_name).lower()
                display_name = "Sport" if cls_clean in ["sport", "0"] else ("Finance" if cls_clean in ["finance", "1"] else str(cls_name))
                probabilities[display_name] = f"{prob * 100:.2f}%"

            status_note = "Berita berhasil diklasifikasikan menggunakan Word2Vec Skip-Gram + Gaussian Naive Bayes."
            return final_label, confidence, probabilities, status_note

        except Exception as e:
            # Fallback jika terjadi error komputasi
            final_label = detected_domain.capitalize()
            confidence = 85.0
            probabilities = {"Sport": "50.00%", "Finance": "50.00%"}
            status_note = f"Klasifikasi fallback berdasarkan pencocokan kata kunci domain ({str(e)})."
            return final_label, confidence, probabilities, status_note
    else:
        # Fallback jika model belum termuat
        final_label = detected_domain.capitalize()
        confidence = 80.0
        probabilities = {"Sport": "N/A", "Finance": "N/A"}
        status_note = "Prediksi fallback berbasis deteksi kata kunci (model ML tidak tersedia)."
        return final_label, confidence, probabilities, status_note


# ==================================================================================================
# 8. PIPELINE TAHAP 6: CONTROLLER & FLASK ROUTING (END-TO-END EXECUTION)
# ==================================================================================================
# Penjelasan Alur:
# - Route GET / : Menampilkan formulir antarmuka input tautan berita.
# - Route POST/GET /classify : Menjalankan pipeline lengkap dari awal hingga akhir:
#     1. Tangkap parameter link URL berita dari request.
#     2. Scraping web untuk mengambil judul dan isi artikel.
#     3. Preprocessing teks (Cleaning, Tokenizing, Stopwords Removal - Tanpa Stemming).
#     4. Ekstraksi fitur vektor dokumen menggunakan Word2Vec Skip-Gram.
#     5. Pengecekan kata kunci domain (Sport vs Finance vs Other).
#     6. Prediksi klasifikasi dan probabilitas dengan Gaussian Naive Bayes.
#     7. Kirim data hasil pemrosesan ke template visual (result.html).
# ==================================================================================================

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Halaman Awal: Menyediakan form input URL berita untuk pengguna.
    """
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        if not url:
            flash("Silakan masukkan URL tautan berita yang valid!", "error")
            return redirect(url_for("index"))
        
        return redirect(url_for("classify", url=url))

    return render_template("index.html", model_error=model_error)


@app.route("/classify", methods=["GET", "POST"])
def classify():
    """
    Endpoint Eksekusi Pipeline:
    URL Berita -> Web Scraping -> Preprocessing -> Vektorisasi Skip-Gram -> Klasifikasi NB -> Render Hasil
    """
    # Langkah 1: Menerima URL berita dari form input (POST) atau query param (GET)
    url = request.args.get("url") or request.form.get("url")
    if not url:
        flash("Silakan masukkan URL tautan berita!", "error")
        return redirect(url_for("index"))

    # Langkah 2: Menjalankan Web Scraping (Tahap 1)
    article_title, article_text = scrape_article_from_url(url)
    if not article_text or article_text.startswith("Gagal"):
        error_message = article_text if article_text.startswith("Gagal") else "Isi artikel berita tidak dapat diekstrak dari tautan yang diberikan."
        return render_template("result.html", 
                               error=error_message, 
                               url=url, 
                               label="Error", 
                               badge_color="red")

    # Langkah 3: Menjalankan Text Preprocessing (Tahap 2 - Tanpa Stemming)
    tokens = preprocess_pipeline(article_text)
    if len(tokens) == 0:
        return render_template("result.html", 
                               error="Teks artikel tidak memiliki kata valid yang tersisa setelah melalui tahap pembersihan (cleansing & stopword removal).", 
                               url=url, 
                               label="Error", 
                               badge_color="red")

    # Langkah 4: Deteksi Domain Kata Kunci (Tahap 4)
    detected_domain, domain_matches = detect_article_domain(tokens)

    # Langkah 5: Vektorisasi Fitur Word2Vec Skip-Gram (Tahap 3)
    vocab_coverage = calculate_vocabulary_coverage(tokens, skipgram_model)
    doc_vector = calculate_document_vector(tokens, skipgram_model)

    # Langkah 6: Klasifikasi dengan Gaussian Naive Bayes (Tahap 5)
    final_label, confidence, probabilities, status_note = predict_article_category(doc_vector, detected_domain)

    # Langkah 7: Render Tampilan Visual Hasil Klasifikasi (Tahap 6)
    return render_template(
        "result.html",
        url=url,
        article_title=article_title,
        article_text=article_text,
        tokens=tokens,
        tokens_str=" ".join(tokens),
        label=final_label,
        confidence=f"{confidence:.2f}",
        probabilities=probabilities,
        matches=domain_matches,
        coverage=f"{vocab_coverage:.2f}",
        status_note=status_note,
        vector_size=vector_size,
        window=window,
        epochs=epochs
    )


# ==================================================================================================
# 9. ENTRY POINT APLIKASI
# ==================================================================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug_mode = os.getenv("FLASK_DEBUG", "True").lower() in ["true", "1", "t"]
    app.run(debug=debug_mode, host="0.0.0.0", port=port)