import os
import re
import pickle
import requests
import numpy as np
from bs4 import BeautifulSoup
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from flask import Flask, render_template, request, redirect, url_for, flash
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "news_classifier_secret_key_ver2")

# ====================================================
# LOAD MODEL & INITIALIZATION
# ====================================================

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
        model_error = f"File model tidak ditemukan di path: {MODEL_PATH}"
except Exception as e:
    model_error = str(e)

# Initialize Stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Stopwords List
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

# Sport Keywords
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

# Finance Keywords
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

# ====================================================
# PREPROCESSING & HELPER FUNCTIONS
# ====================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text):
    return text.split()

def remove_stopwords(tokens):
    return [token for token in tokens if token not in STOPWORDS]

def stemming(tokens):
    return [stemmer.stem(token) for token in tokens]

def preprocess_text(text):
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens_no_stop = remove_stopwords(tokens)
    stemmed_tokens = stemming(tokens_no_stop)
    return stemmed_tokens

def scrape_url(url):
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Try title extraction
        title = ""
        title_tag = soup.find("h1") or soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()

        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
        return title, text.strip()
    except Exception as e:
        return "", f"Error HTTP/Scraping: {e}"

def document_vector(tokens, model):
    vectors = []
    if model is not None and hasattr(model, "wv"):
        for token in tokens:
            if token in model.wv:
                vectors.append(model.wv[token])
    
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

def calculate_vocabulary_coverage(tokens, model):
    if len(tokens) == 0 or model is None or not hasattr(model, "wv"):
        return 0.0
    known_tokens = [token for token in tokens if token in model.wv]
    return (len(known_tokens) / len(tokens)) * 100.0

def get_domain_matches(tokens):
    token_set = set(tokens)
    sport_matches = token_set.intersection(SPORT_KEYWORDS)
    finance_matches = token_set.intersection(FINANCE_KEYWORDS)
    strong_sport_matches = token_set.intersection(STRONG_SPORT_KEYWORDS)
    strong_finance_matches = token_set.intersection(STRONG_FINANCE_KEYWORDS)
    return sport_matches, finance_matches, strong_sport_matches, strong_finance_matches

def detect_domain(tokens):
    sport_matches, finance_matches, strong_sport_matches, strong_finance_matches = get_domain_matches(tokens)

    sport_score = len(sport_matches)
    finance_score = len(finance_matches)
    strong_sport_score = len(strong_sport_matches)
    strong_finance_score = len(strong_finance_matches)

    matches_data = {
        "sport": sorted(list(sport_matches)),
        "finance": sorted(list(finance_matches)),
        "strong_sport": sorted(list(strong_sport_matches)),
        "strong_finance": sorted(list(strong_finance_matches))
    }

    if sport_score == 0 and finance_score == 0:
        return "other", matches_data

    if strong_sport_score >= 1 and strong_sport_score > strong_finance_score:
        return "sport", matches_data

    if strong_finance_score >= 1 and strong_finance_score > strong_sport_score:
        return "finance", matches_data

    if sport_score >= 2 and sport_score > finance_score:
        return "sport", matches_data

    if finance_score >= 2 and finance_score > sport_score:
        return "finance", matches_data

    return "other", matches_data


# ====================================================
# FLASK ROUTES
# ====================================================

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        if not url:
            flash("Silakan masukkan URL berita yang valid!", "error")
            return redirect(url_for("index"))
        
        return redirect(url_for("classify", url=url))

    return render_template("index.html", model_error=model_error)

@app.route("/classify", methods=["GET", "POST"])
def classify():
    url = request.args.get("url") or request.form.get("url")
    if not url:
        flash("Silakan masukkan URL berita!", "error")
        return redirect(url_for("index"))

    # 1. Scrape Text
    article_title, article_text = scrape_url(url)
    if not article_text or article_text.startswith("Error"):
        error_msg = article_text if article_text.startswith("Error") else "Teks berita tidak berhasil diambil dari URL."
        return render_template("result.html", 
                               error=error_msg, 
                               url=url, 
                               label="Error", 
                               badge_color="red")

    # 2. Preprocess Text
    tokens = preprocess_text(article_text)
    if len(tokens) == 0:
        return render_template("result.html", 
                               error="Teks berita tidak memiliki kata valid yang dapat diproses setelah preprocessing.", 
                               url=url, 
                               label="Error", 
                               badge_color="red")

    # 3. Domain Detection (Sport / Finance / Other)
    detected_domain, matches_data = detect_domain(tokens)

    # 4. Vectorize & Vocabulary Coverage
    coverage = calculate_vocabulary_coverage(tokens, skipgram_model)
    doc_vec = document_vector(tokens, skipgram_model)

    # If domain is unknown / other
    if detected_domain == "other":
        final_label = "Other"
        confidence = 100.0
        probabilities = {"Sport": "0.00%", "Finance": "0.00%", "Other": "100.00%"}
        status_note = "Berita ini dikategorikan sebagai 'Other' (Di luar Sport & Finance) karena tidak ditemukan indikasi keyword domain yang cukup."
    else:
        # Perform Naive Bayes Prediction
        if naive_bayes is not None and scaler is not None and not np.all(doc_vec == 0):
            try:
                doc_vec_scaled = scaler.transform([doc_vec])
                raw_pred = naive_bayes.predict(doc_vec_scaled)[0]
                raw_probs = naive_bayes.predict_proba(doc_vec_scaled)[0]
                
                raw_pred_str = str(raw_pred).lower()
                if raw_pred_str in ["sport", "0"]:
                    final_label = "Sport"
                elif raw_pred_str in ["finance", "1"]:
                    final_label = "Finance"
                else:
                    final_label = str(raw_pred).capitalize()
                
                max_prob = float(np.max(raw_probs)) * 100.0
                confidence = max_prob
                
                # Format probability distribution
                classes = naive_bayes.classes_
                probabilities = {}
                for cls, prob in zip(classes, raw_probs):
                    cls_str = str(cls).lower()
                    display_cls = "Sport" if cls_str in ["sport", "0"] else ("Finance" if cls_str in ["finance", "1"] else str(cls))
                    probabilities[display_cls] = f"{prob * 100:.2f}%"
                
                status_note = f"Berita berhasil diklasifikasikan menggunakan Word2Vec Skip-Gram + Gaussian Naive Bayes."
            except Exception as e:
                final_label = detected_domain.capitalize()
                confidence = 85.0
                probabilities = {"Sport": "50.00%", "Finance": "50.00%"}
                status_note = f"Klasifikasi berdasarkan keyword domain match ({e})."
        else:
            final_label = detected_domain.capitalize()
            confidence = 80.0
            probabilities = {"Sport": "N/A", "Finance": "N/A"}
            status_note = "Prediksi fallback berdasarkan deteksi keyword kata kunci."

    return render_template("result.html",
                           url=url,
                           article_title=article_title,
                           article_text=article_text,
                           tokens=tokens,
                           tokens_str=" ".join(tokens),
                           label=final_label,
                           confidence=f"{confidence:.2f}",
                           probabilities=probabilities,
                           matches=matches_data,
                           coverage=f"{coverage:.2f}",
                           status_note=status_note,
                           vector_size=vector_size,
                           window=window,
                           epochs=epochs)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() in ["true", "1", "t"]
    app.run(debug=debug, host="0.0.0.0", port=port)