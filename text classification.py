from sklearn.datasets import fetch_20newsgroups
import fasttext
import re
import os

MODEL_PATH = "fasttext_model_20news.bin"
TRAIN_TXT = "train_data.txt"
TEST_TXT = "test_data.txt"

# --- Text cleaning ---
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text.lower()

# --- Save data in FastText format ---
def save_to_file(data, labels, filename):
    with open(filename, 'w', encoding="utf-8") as f:
        for text, label in zip(data, labels):
            cleaned_text = clean_text(text)
            f.write(f"__label__{label} {cleaned_text}\n")

# --- Ensure train/test files exist (create if missing) ---
def ensure_data_files():
    if not (os.path.exists(TRAIN_TXT) and os.path.exists(TEST_TXT)):
        print("Preparing 20 Newsgroups data (train/test)...")
        ng_train = fetch_20newsgroups(subset='train')
        ng_test = fetch_20newsgroups(subset='test')
        save_to_file(ng_train.data, ng_train.target, TRAIN_TXT)
        save_to_file(ng_test.data, ng_test.target, TEST_TXT)

# --- Train model only if missing ---
def get_or_train_model():
    if os.path.exists(MODEL_PATH):
        print(f"✅ Found existing model: {MODEL_PATH}. Loading...")
        return fasttext.load_model(MODEL_PATH)
    else:
        print("⚡ No model found. Training a new one...")
        ensure_data_files()
        model = fasttext.train_supervised(
            TRAIN_TXT, epoch=35, lr=1.0, wordNgrams=3, verbose=1, minCount=1
        )
        model.save_model(MODEL_PATH)
        print(f"💾 Model saved to {MODEL_PATH}")
        return model

# --- Build label map (index -> category name) ---
def build_label_map():
    # We fetch just to get target_names; cheap and keeps mapping robust.
    ng_train = fetch_20newsgroups(subset='train')
    return {f"__label__{i}": cat for i, cat in enumerate(ng_train.target_names)}

# --- Pretty print evaluation ---
def print_results(N, p, r):
    print(f"N\t\t{N}")
    print(f"Precision\t{p:.3f}")
    print(f"Recall\t\t{r:.3f}")

# ---------- Main ----------
# Make sure test file exists for evaluation
ensure_data_files()

# Load or train model
model = get_or_train_model()

# Evaluate on test set
print("\nEvaluation on test set:")
print_results(*model.test(TEST_TXT))

# Map labels to human-readable category names
label_map = build_label_map()

# Hardcoded examples
examples = [
    "NASA announced a new discovery about black holes.",
    "3D graphics rendering has improved with new GPU technology.",
    "Christianity and atheism are often debated in philosophy.",
    "The latest version of Windows improves system security.",
    "The hockey team won their last game with a great score."
]

print("\nClassifying hardcoded examples:\n")
for text in examples:
    cleaned = clean_text(text)
    label, prob = model.predict(cleaned)
    category = label_map.get(label[0], label[0])
    print(f"Text: {text}")
    print(f" → Classified as: {category} (p={prob[0]:.2f})\n")
