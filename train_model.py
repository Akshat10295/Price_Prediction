import pandas as pd
import numpy as np
import requests
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import make_scorer
import os.path 

# --- 0. Define the Competition Metric: SMAPE ---
def smape_metric(y_true, y_pred):
    """Calculates SMAPE score"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    smape_val = np.mean(
        np.divide(2 * np.abs(y_pred - y_true), 
                  denominator, 
                  out=np.zeros_like(denominator, dtype=float), 
                  where=denominator!=0)
    )
    return 100 * smape_val

# --- 1. Load Data ---
print("Loading data...")
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
except FileNotFoundError:
    print("Error: 'train.csv' or 'test.csv' not found.")
    print("Please place your training and testing files in the same directory.")
    exit()

# --- MODIFICATION START (FOR QUICK TESTING) ---
# To test your script, uncomment the 3 lines below
# print("!!! WARNING: RUNNING ON A SMALL SAMPLE OF 1000 ROWS !!!")
# train_df = train_df.head(1000)
# test_df = test_df.head(1000)
# --- MODIFICATION END ---

print("Data loaded successfully.")
print(f"Training data shape: {train_df.shape}")
print(f"Testing data shape: {test_df.shape}")

# Handle potential missing text data
train_df['catalog_content'] = train_df['catalog_content'].fillna('missing')
test_df['catalog_content'] = test_df['catalog_content'].fillna('missing')


# --- 2. Text Feature Engineering (Embeddings) ---
# Check if pre-computed text embeddings exist
if os.path.exists('X_train_text.npy') and os.path.exists('X_test_text.npy'):
    print("Loading text features from disk...")
    X_train_text = np.load('X_train_text.npy')
    X_test_text = np.load('X_test_text.npy')
else:
    print("\n--- Step 2: Generating Text Embeddings ---")
    text_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    X_train_text = text_model.encode(
        train_df['catalog_content'].tolist(), 
        show_progress_bar=True,
        batch_size=32 
    )
    X_test_text = text_model.encode(
        test_df['catalog_content'].tolist(), 
        show_progress_bar=True,
        batch_size=32
    )
    
    # Save the generated features to disk
    print("Saving text features to disk...")
    np.save('X_train_text.npy', X_train_text)
    np.save('X_test_text.npy', X_test_text)

print("Text embeddings complete.")


# --- 3. Image Feature Engineering (Embeddings) ---
# Check if pre-computed image embeddings exist
if os.path.exists('X_train_img.npy') and os.path.exists('X_test_img.npy'):
    print("Loading image features from disk...")
    X_train_img = np.load('X_train_img.npy')
    X_test_img = np.load('X_test_img.npy')
else:
    print("\n--- Step 3: Generating Image Embeddings ---")
    resnet = models.resnet50(pretrained=True)
    resnet.fc = torch.nn.Identity()
    resnet.eval() 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resnet.to(device)
    print(f"Using device: {device} for image embeddings")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    EMBEDDING_DIM = 2048 

    def get_image_embedding(url):
        """Downloads and processes a single image URL to get its embedding."""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status() 
            
            img = Image.open(BytesIO(response.content)).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                emb = resnet(img_t)
            
            return emb.cpu().numpy().flatten()
        except Exception as e:
            return np.zeros(EMBEDDING_DIM)

    tqdm.pandas(desc="Processing Images")

    print("Generating TRAIN image embeddings (this will take a very long time)...")
    X_train_img_series = train_df['image_link'].progress_apply(get_image_embedding)

    print("Generating TEST image embeddings (this will also take a long time)...")
    X_test_img_series = test_df['image_link'].progress_apply(get_image_embedding)

    X_train_img = np.stack(X_train_img_series.values)
    X_test_img = np.stack(X_test_img_series.values)
    
    # Save the generated features to disk
    print("Saving image features to disk...")
    np.save('X_train_img.npy', X_train_img)
    np.save('X_test_img.npy', X_test_img)

print("Image embeddings complete.")


# --- 4. Feature Concatenation ---
print("\n--- Step 4: Combining Text and Image Features ---")
X_train = np.concatenate([X_train_text, X_train_img], axis=1)
X_test = np.concatenate([X_test_text, X_test_img], axis=1)

print(f"Combined training features shape: {X_train.shape}")
print(f"Combined testing features shape: {X_test.shape}")

y_train = np.log1p(train_df['price'])


# --- 5. Model Training & Validation ---
print("\n--- Step 5: Model Training & Validation ---")
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)

print(f"Training on {X_train_split.shape[0]} samples, validating on {X_val_split.shape[0]} samples.")

model = GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    random_state=42,
    validation_fraction=0.1,
    n_iter_no_change=20,
    tol=0.01,
    verbose=1
)

model.fit(X_train_split, y_train_split)

# --- 6. Model Evaluation (Validation) ---
print("\n--- Step 6: Model Evaluation ---")
y_val_pred_log = model.predict(X_val_split)

y_val_pred = np.expm1(y_val_pred_log)
y_val_true = np.expm1(y_val_split)

y_val_pred = y_val_pred.clip(min=0)

val_smape = smape_metric(y_val_true, y_val_pred)
print(f"\n========================================")
print(f"Validation SMAPE Score: {val_smape:.4f}")
print(f"========================================")


# --- 7. Final Training & Prediction for Submission ---
print("\n--- Step 7: Training Final Model ---")
print("Retraining model on ALL training data...")
final_model = GradientBoostingRegressor(
    n_estimators=model.n_estimators_, 
    learning_rate=0.05,
    max_depth=7,
    random_state=42,
    verbose=1
)
final_model.fit(X_train, y_train)

print("Making predictions on test set...")
y_test_pred_log = final_model.predict(X_test)

y_test_pred = np.expm1(y_test_pred_log)
y_test_pred = y_test_pred.clip(min=0)

output_df = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price': y_test_pred
})

output_df.to_csv('test_out.csv', index=False)

print("\n--- All Done! ---")
print("Submission file 'test_out.csv' created successfully.")