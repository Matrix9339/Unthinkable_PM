# advanced_app_streamlit.py
import streamlit as st
from PIL import Image
from model_loader import products, product_vectors, get_image_embedding, cosine_similarity
import os
import pandas as pd

# ---------------- Config ----------------
st.set_page_config(page_title="Visual Product Search", layout="wide")
st.title("Visual Product Search")

# ---------------- Helper Functions ----------------
@st.cache_data(show_spinner=False)
def find_similar(uploaded_path, threshold=0.5, top_k=5):
    query_vec = get_image_embedding(uploaded_path)
    if not query_vec:
        return []

    sims = []
    for item in product_vectors:
        if "vector" not in item:
            continue
        sim = cosine_similarity(query_vec, item["vector"]).item()
        sims.append((item["product"], sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    results = [x for x in sims if x[1] >= threshold]
    if not results:
        results = sims[:top_k]
    return results[:top_k]


def display_results(results):
    if not results:
        st.info("No products found. Try adjusting category or similarity threshold.")
        return

    # Inject CSS for responsive grid layout
    st.markdown(
        """
        <style>
        .result-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 10px;
        }
        .card {
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 12px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .card img {
            width: 100%;
            border-radius: 8px;
            height: auto;
            object-fit: cover;
        }
        .card h4 {
            margin: 8px 0 4px 0;
            font-size: 16px;
            font-weight: 600;
        }
        .card p {
            margin: 2px 0;
            font-size: 14px;
            color: #333;
        }
        @media (max-width: 768px) {
            .card { padding: 10px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Build HTML grid
    html = '<div class="result-grid">'
    for product, score in results:
        html += f"""
        <div class="card">
            <img src="{product['thumbnail']}" alt="{product['title']}">
            <h4>{product['title']}</h4>
            <p>Category: {product['category']}</p>
            <p>Price: ₹{product['price']}</p>
            <p>Similarity: {round(score, 2)}</p>
        </div>
        """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ---------------- Sidebar Filters ----------------
st.sidebar.header("Filters")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
image_url = st.sidebar.text_input("Or Image URL", placeholder="https://...")
categories = sorted({(p.get('category') or '').strip() for p in products if p.get('category')})
selected_categories = st.sidebar.multiselect(
    "Select Category(s)", ["all"] + categories, default=["all"]
)
similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.01)
top_k = st.sidebar.slider("Top K Results", 1, 20, 5)

# ---------------- Search Button ----------------
if st.sidebar.button("Search"):
    uploaded_path = None
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        uploaded_path = f"temp_{uploaded_file.name}"
        img.save(uploaded_path)
    elif image_url:
        uploaded_path = image_url
    else:
        st.warning("Please upload a file or enter an image URL")

    if uploaded_path:
        st.subheader("Uploaded Image")
        try:
            if uploaded_file:
                st.image(img, width=250)
            elif image_url:
                st.image(image_url, width=250)
        except Exception:
            st.warning("Unable to display the uploaded image.")

        results = find_similar(uploaded_path, threshold=similarity_threshold, top_k=top_k)

        if "all" not in selected_categories:
            results = [(p, s) for p, s in results if p.get('category') in selected_categories]

        st.subheader("Search Results")
        display_results(results)

        if results:
            df = pd.DataFrame([
                {
                    "title": p['title'],
                    "category": p['category'],
                    "price": p['price'],
                    "similarity": round(s, 3),
                }
                for p, s in results
            ])
            st.download_button(
                "Download Results as CSV",
                df.to_csv(index=False),
                "results.csv",
                "text/csv",
            )
