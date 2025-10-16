import torch
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
import requests
import os
import json

# Config 
PRODUCT_JSON = "products.json"
VECTOR_JSON = "product_vectors.json"
MODEL_FILE = "mobilenetv3_small.pt"

device = "cpu"
print("✅ Device:", device)

# Load MobileNetV3 Small
def load_model():
    model = models.mobilenet_v3_small(weights=None)
    if os.path.exists(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
        print("✅ Loaded MobileNetV3 small from local file")
    else:
        print("⚠️ MODEL FILE NOT FOUND! Please run locally to download and commit 'mobilenetv3_small.pt'")
        raise FileNotFoundError("mobilenetv3_small.pt not found")

    model.classifier = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model

model = load_model()

# Preprocessing 
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Helper Functions
def get_image_embedding(path_or_url):
    """Convert image (path or URL) to embedding vector."""
    try:
        if str(path_or_url).startswith("http"):
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(path_or_url, headers=headers, timeout=10)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            img = Image.open(path_or_url).convert("RGB")

        img_input = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(img_input)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze(0).tolist()
    except Exception as e:
        print(f"❌ Error embedding {path_or_url}: {e}")
        return None


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two embeddings."""
    a = torch.tensor(vec1)
    b = torch.tensor(vec2)
    return torch.dot(a, b) / (a.norm() * b.norm() + 1e-8)


# Load Products & Precomputed Vectors 
def load_json_file(path):
    if not os.path.exists(path):
        print(f"❌ File '{path}' not found!")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                print(f"❌ '{path}' should contain a list, got {type(data)}")
                return []
            return data
    except Exception as e:
        print(f"❌ Error reading '{path}': {e}")
        return []

products = load_json_file(PRODUCT_JSON)
product_vectors = load_json_file(VECTOR_JSON)

if not products:
    raise ValueError("❌ 'products.json' is empty or invalid! Add valid products before deploying.")

if not product_vectors:
    raise ValueError("❌ 'product_vectors.json' is empty or invalid! Precompute vectors before deploying.")

print(f"✅ Loaded {len(products)} products and {len(product_vectors)} precomputed vectors.")
