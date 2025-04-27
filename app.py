import streamlit as st
st.set_page_config(page_title="🔬 Scientific Abstract Analyzer", layout="wide")

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import joblib
import datetime

# 🎨 Özel CSS ile modern butonlar ve renkler
st.markdown("""
    <style>
        body {
            background-color: #1f1f1f;
            color: #e0e0e0;
        }

        .stButton>button {
            background: linear-gradient(to right, #8A2387, #E94057, #F27121);
            color: white;
            border-radius: 12px;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
        }

        .stTextArea textarea {
            background-color: #2c2c2c;
            color: white;
            border: 1px solid #444;
            border-radius: 10px;
            padding: 12px;
            font-size: 16px;
        }

        .stSlider>div>label {
            color: #e0e0e0;
        }
    </style>
""", unsafe_allow_html=True)

# 📦 Verileri ve modelleri yükle
@st.cache_resource
def load_everything():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    xgb_model = joblib.load("domain_final_model.pkl")
    embeddings = np.load("filtered_abstracts_embeddings.npy")
    df = pd.read_csv("processed_abstracts.csv")
    return model, xgb_model, embeddings, df

specter_model, final_model, embeddings, df = load_everything()

# 🧠 Etiketleri dönüştür
le = LabelEncoder()
le.fit(df["Domain"])

# 🔍 Ana analiz fonksiyonu
def analyze_abstract_v2(user_input, top_k=5, threshold=0.35):
    try:
        if not isinstance(user_input, str) or len(user_input.strip()) < 20:
            return {"error": "❌ Lütfen geçerli bir makale özeti girin."}

        # ➤ Kullanıcı girdisini embed et
        user_vec = specter_model.encode([user_input])

        # ➤ Domain tahmini
        probas = final_model.predict_proba(user_vec)[0]
        domain_pred_idx = np.argmax(probas)
        domain_pred = le.inverse_transform([domain_pred_idx])[0]
        confidence = float(probas[domain_pred_idx])

        # ➤ Çoklu etiket tahmini (multi-label)
        multi_domains = [
            le.inverse_transform([i])[0]
            for i, p in enumerate(probas) if p >= threshold
        ]

        # ➤ En benzer özetleri bul
        domain_df = df[df["Domain"] == domain_pred].copy()
        domain_embeds = embeddings[domain_df.index]
        sims = cosine_similarity(user_vec, domain_embeds)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        top_examples = domain_df.iloc[top_idx]
        top_scores = sims[top_idx]

        matches = []
        for i in range(top_k):
            matches.append({
                "rank": i + 1,
                "similarity_score": round(float(top_scores[i]), 4),
                "matched_domain": top_examples.iloc[i]["Domain"],
                "matched_abstract": top_examples.iloc[i]["Abstract"][:500]
            })

        return {
            "predicted_domain": domain_pred,
            "confidence_score": round(confidence, 4),
            "multi_label_domains": multi_domains,
            "top_k_matches": matches
        }

    except Exception as e:
        return {"error": f"⚠️ Hata oluştu: {str(e)}"}

# 🎛️ Streamlit Arayüzü
st.title("🧠 Scientific Abstract Analyzer")
st.markdown("Bu uygulama, makale özetlerini anlayarak biyomedikal domain tahmini yapar ve benzer makaleleri bulur.")

user_input = st.text_area("✍️ Lütfen bilimsel makale özetinizi girin:", height=200)
top_k = st.slider("🔍 Gösterilecek en benzer özet sayısı:", 1, 10, 5)
threshold = st.slider("📊 Çoklu domain eşiği:", 0.0, 1.0, 0.35)

if st.button("Analyze"):
    result = analyze_abstract_v2(user_input, top_k=top_k, threshold=threshold)
    
    # 1. Güven Skoru Görselleştirme: Renkli Progress Bar
    if "error" in result:
        st.error(result["error"])
    else:
        st.markdown(f"### ✅ Tahmin Edilen Domain: `{result['predicted_domain']}`")
        
        # Progress bar rengi güven skoruna göre dinamikleşebilir
        progress = result['confidence_score']
        if progress > 0.75:
            st.progress(progress, text="Yüksek Güven")
        elif progress > 0.50:
            st.progress(progress, text="Orta Güven")
        else:
            st.progress(progress, text="Düşük Güven")
        
        st.markdown(f"**🔒 Güven Skoru: {result['confidence_score'] * 100:.2f}%**")
        
        # 2. Kullanıcı Girdisi ve Sonuçlarını Loglama (feedback.log)
        import datetime
        def log_feedback(user_input, result):
            with open("feedback.log", "a", encoding="utf-8") as f:
                f.write(f"\n--- {datetime.datetime.now()} ---\n")
                f.write(f"Input: {user_input}\n")
                f.write(f"Predicted: {result['predicted_domain']} | Confidence: {result['confidence_score']}\n")
                f.write(f"Multi-label: {', '.join(result['multi_label_domains'])}\n")
                f.write("\n")

        log_feedback(user_input, result)

        st.success(f"✅ Tahmin Edilen Domain: **{result['predicted_domain']}** (Güven: {result['confidence_score']})")
        st.info(f"🏷️ Çoklu Domain Etiketleri: {', '.join(result['multi_label_domains'])}")
        st.markdown("### 📄 En Benzer Makale Özetleri:")
        
        # 3. Benzer Makale Eşleşmeleri için Accordion (st.expander)
        for match in result["top_k_matches"]:
            with st.expander(f"**{match['rank']}. (Benzerlik: {match['similarity_score']})**"):
                st.markdown(f"**Domain:** {match['matched_domain']}")
                st.markdown(f"{match['matched_abstract']}...")

# 🧠 Kullanıcı geri bildirimi için text input alanı
user_feedback = st.text_input("📝 Geri bildiriminizi buraya yazın:")

if user_feedback:
    st.write(f"Teşekkürler! Geri bildiriminiz: {user_feedback}")
    # Burada istersen loglama veya veri gönderme işlemi yapılabilir

