Visual Product Search
A deep learning-powered visual search engine that finds similar products using MobileNetV3. Upload an image to discover visually similar items from the catalog.

Live Application:
Try it here: https://unthinkable-task.streamlit.app/

Features:
Visual Search: Upload images or use URLs
Real-time Matching: Cosine similarity with precomputed vectors
Smart Filters: Category and similarity threshold filters
Export Results: Download results as CSV

Tech Stack:
Streamlit (Frontend)
PyTorch & MobileNetV3 (Computer Vision)
Cosine Similarity (Vector Search)


Approach:
Feature Extraction: MobileNetV3 processes product images into 1024-dimensional embedding vectors, capturing visual patterns like shape, color, and texture.
Precomputation: All product images are pre-processed into vectors stored in product_vectors.json, enabling real-time search without model inference during queries.
Similarity Matching: Cosine similarity compares query image embeddings against precomputed vectors, ranking products by visual resemblance.
Efficient Deployment: The Streamlit interface provides intuitive upload options (file/URL) with filtering by category and similarity thresholds. The system runs entirely on CPU, making it cost-effective for deployment.
This solution balances accuracy with performance, using a lightweight model suitable for web deployment while maintaining good visual recognition capabilities. The precomputation strategy ensures fast response times, crucial for user experience in e-commerce applications.
Key Innovation: Offloading heavy computation to preprocessing phase, enabling instant visual search comparable to commercial platforms within constrained resources.
