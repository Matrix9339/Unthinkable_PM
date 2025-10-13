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
    if results:
        n_cols = 4
        cols = st.columns(n_cols)
        for idx, (product, score) in enumerate(results):
            col = cols[idx % n_cols]
            with col:
                st.image(product["thumbnail"], use_container_width=True)
                st.markdown(f"**{product['title']}**")
                st.markdown(f"Category: {product['category']}")
                st.markdown(f"Price: ₹{product['price']}")
                st.markdown(f"Similarity: {round(score,2)}")
    else:
        st.info("No products found. Try adjusting category or similarity threshold.")

# ---------------- Sidebar Filters ----------------
st.sidebar.header("Filters")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
image_url = st.sidebar.text_input("Or Image URL", placeholder="https://...")
categories = sorted({(p.get('category') or '').strip() for p in products if p.get('category')})
selected_categories = st.sidebar.multiselect("Select Category(s)", ["all"] + categories, default=["all"])
similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.01)
top_k = st.sidebar.slider("Top K Results", 1, 20, 5)

# ---------------- Search Button ----------------
if st.sidebar.button("Search"):
    # Handle uploaded file or URL
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        uploaded_path = f"temp_{uploaded_file.name}"
        img.save(uploaded_path)
    elif image_url:
        uploaded_path = image_url
    else:
        st.warning("Please upload a file or enter an image URL")
        uploaded_path = None

    if uploaded_path:
        # Show uploaded image
        st.subheader("Uploaded Image")
        if uploaded_file:
            st.image(img, width=250)
        elif image_url:
            st.image(image_url, width=250)

        # Find similar products
        results = find_similar(uploaded_path, threshold=similarity_threshold, top_k=top_k)

        # Filter by selected categories
        if "all" not in selected_categories:
            results = [(p, s) for p, s in results if p.get('category') in selected_categories]

        st.subheader("Search Results")
        display_results(results)

        # Optional: Download results as CSV
        if results:
            df = pd.DataFrame([{
                "title": p['title'],
                "category": p['category'],
                "price": p['price'],
                "similarity": round(s, 3)
            } for p, s in results])
            st.download_button("Download Results CSV", df.to_csv(index=False), "results.csv", "text/csv")
