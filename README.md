# Sneaker Authentic vs Fake Classification

A deep learning system for detecting counterfeit branded sneakers using a **multimodal Vision Transformer (ViT)** approach that combines image analysis, brand recognition, and price information to determine authenticity.

## Project Overview

This project implements a binary classifier that predicts whether a sneaker listing is **Authentic (1)** or **Fake (0)**. The model leverages:

1. **Image Features**: Vision Transformer (ViT-Base) extracts visual features from sneaker images, learning to recognize authentic vs. fake characteristics
2. **Brand Information**: Embedding layer maps brand names to learned representations, helping the model understand brand-specific authentic patterns
3. **Price Information**: Price is used as a decision factor (generally, authentic sneakers have higher prices than counterfeits)

### Implementation (Colab Notebook)

[https://colab.research.google.com/drive/1v31ueZuGqZVygRCz74EauxcQ_uSfQ06y?usp=sharing](Link to Colab Notebook)

### Model Architecture

The model (`ViTMultiHeadImproved`) consists of:

1. **Image Backbone**: Pretrained ViT-Base-Patch16-224 from `timm` library
   - Extracts 768-dimensional image features
   - **Partial Fine-Tuning**: First 2 transformer layers are frozen, remaining 10 layers are fine-tuned
   - Projected to 512-dimensional embeddings (with Dropout 0.15 and LayerNorm)

2. **Brand Embedding**: Learnable embedding layer (64 dimensions) for brand classification
   - Maps brand names (Adidas, Nike, Jordan, Puma, Reebok) to dense vectors

3. **Price Projection**: MLP that projects price (log-normalized and standardized) to 64 dimensions
   - Architecture: Linear(1 → 64) + ReLU + Dropout(0.1) + LayerNorm

4. **Multi-Head Output**:
   - **Authenticity Head**: Combines image (512) + brand (64) + price (64) → predicts Authentic/Fake
     - Architecture: Linear(640 → 512) + ReLU + Dropout(0.15) → Linear(512 → 256) + ReLU + Dropout(0.25) → Linear(256 → 1)
   - **Brand Classification Head**: Uses image features alone → predicts brand (auxiliary task)
     - Architecture: Linear(512 → 256) + ReLU + Dropout(0.1) → Linear(256 → 5 brands)
   - **Contrastive Learning Branch**: Projects image embeddings for supervised contrastive loss
     - Architecture: Linear(512 → 512) + ReLU + LayerNorm

### Training Strategy

The model is trained with three loss components:

1. **Primary Loss**: Binary Cross-Entropy with Logits (BCEWithLogitsLoss) for authenticity prediction
   - Standard BCE loss (no focal loss, no label smoothing)
2. **Auxiliary Loss**: Cross-Entropy for brand classification (weight: 0.3)
   - Helps the model learn brand-specific visual patterns
3. **Contrastive Loss**: Supervised contrastive loss for brand-based image clustering (weight: 0.03)
   - Temperature: 0.1 (sharper contrast)
   - Encourages images of the same brand to cluster together

**Total Loss Formula**:
```
Total Loss = Authenticity Loss + 0.3 × Brand Loss + 0.03 × Contrastive Loss
```

This multi-task learning approach helps the model learn better image representations by:
- Forcing the model to identify brands from images (like catches spelling mistakes in logos)
- Encouraging similar images of the same brand to cluster together
- Using price as additional signal for authenticity
- **Partial Fine-Tuning**: Freezing early layers preserves general features while fine-tuning later layers for domain-specific patterns

---

## Dataset Structure

The project requires a specific folder structure for training and testing data.

### Root Directory Structure

```
Counterfeit-Product-Image-Detection/
├── data/
│   └── sneakers/
│       ├── authentic/          # Training: Authentic sneaker images
│       │   ├── adidas/
│       │   │   ├── 1.png
│       │   │   ├── 2.jpg
│       │   │   └── ...
│       │   ├── nike/
│       │   │   ├── 3.png
│       │   │   └── ...
│       │   ├── jordan/
│       │   ├── puma/
│       │   └── reebok/
│       ├── fake/               # Training: Counterfeit sneaker images
│       │   ├── adidas/
│       │   │   ├── 1.png
│       │   │   ├── 2.jpg
│       │   │   └── ...
│       │   ├── nike/
│       │   ├── jordan/
│       │   ├── puma/
│       │   └── reebok/
│       └── test/               # Testing: Flat structure (all images directly here)
│           ├── test_img1.jpg
│           ├── test_img2.png
│           └── ...
├── Counterfeit_product_data - Adidas.csv
├── Counterfeit_product_data - Nike.csv
├── Counterfeit_product_data - Jordan .csv
├── Counterfeit_product_data - Puma.csv
├── Counterfeit_product_data - Reebok.csv
├── test.csv                    # Test set CSV
└── training3 (1).ipynb         # Main training notebook
```

### Training Image Structure

**Path Pattern**: `data/sneakers/{authentic|fake}/{brand_lowercase}/{image_filename}`

**Example Paths**:
- Authentic Adidas: `data/sneakers/authentic/adidas/1.png`
- Fake Nike: `data/sneakers/fake/nike/nike_f1.png`
- Authentic Jordan: `data/sneakers/authentic/jordan/jordan_a5.jpg`

**Notes**:
- Brand folder names must be **lowercase** (e.g., `adidas`, not `Adidas`)
- Images can be in any common format (`.png`, `.jpg`, `.jpeg`)
- Each brand should have separate folders under both `authentic/` and `fake/`

### Test Image Structure

**Path Pattern**: `data/sneakers/test/{image_filename}` (flat structure)

**Example**: `data/sneakers/test/test_img1.jpg`

**Note**: Test images are stored in a flat structure (no brand subfolders), and the brand information comes from the CSV file.

---

## CSV File Format

### Training CSV Files

One CSV file per brand with the naming convention: `Counterfeit_product_data - {Brand}.csv`

**Required Columns**:
- `sln`: Serial number (optional, for reference)
- `Image Name`: Filename of the image (e.g., `1.png`, `2.jpg`)
- `Brand`: Brand name (e.g., `Adidas`, `Nike`, `Jordan`, `Puma`, `Reebok`)
- `Price`: Product price as a float (e.g., `99.00`, `159.99`)
- `Authentic`: Binary label (`1` for authentic, `0` for fake)

**Example CSV Content**:
```csv
sln,Image Name,Brand,Price,Authentic
1,1.png,Adidas,39.00,0
2,2.jpg,Adidas,99.00,0
3,3.png,Adidas,29.00,0
4,4.jpg,Adidas,149.99,1
5,5.png,Adidas,199.99,1
```

**Important**:
- The `Image Name` must match the actual filename in the corresponding folder
- Brand names are case-sensitive in CSV but converted to lowercase for folder paths
- All training CSVs are automatically merged during training
- The script excludes any CSV file starting with "test" (case-insensitive)

### Test CSV File

File name: `test.csv`

**Format**: Same as training CSV, but images are located in `data/sneakers/test/` (flat structure)

**Example**:
```csv
sln,Image Name,Brand,Price,Authentic
1,test1.jpg,Adidas,45.00,0
2,test2.png,Nike,89.99,1
3,test3.jpg,Jordan,199.99,1
```

**Note**: The `Authentic` column is optional for inference-only use. If present, it's used to compute test metrics (accuracy, AUC).

---

## Training Process

### Prerequisites

1. **Google Colab** (recommended) with GPU enabled
2. **Required Packages**:
   - `torch`, `torchvision`
   - `timm` (for Vision Transformer)
   - `pandas`, `numpy`
   - `scikit-learn`
   - `PIL` (Pillow)
   - `tqdm`

### Training Configuration

Key hyperparameters (editable in Cell 2):

```python
BATCH_SIZE = 16
IMG_SIZE = 224
NUM_EPOCHS = 20
LR = 2e-4
WEIGHT_DECAY = 1e-4
BACKBONE = 'vit_base_patch16_224'
FREEZE_LAYERS = 2  # Freeze first 2 transformer layers, train remaining 10
BRAND_EMBED_DIM = 64
OUT_DIM = 512
USE_CONTRASTIVE = True
CONTRASTIVE_TEMPERATURE = 0.1
AUX_BRAND_LOSS = True
BRAND_LOSS_WEIGHT = 0.3
CONTRASTIVE_LOSS_WEIGHT = 0.03
USE_FOCAL_LOSS = False
USE_MIXUP = False
LABEL_SMOOTHING = 0.0
PREDICTION_THRESHOLD = 0.4  # Lower threshold to reduce false positives
USE_EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 7
```

**Data Augmentation** (minimal to preserve authentic details):
- Resize to 224×224
- Random horizontal flip (50% probability)
- Minimal rotation (±5 degrees)
- Minimal color jitter (brightness/contrast/saturation: 0.05)
- **No** random crop, perspective, affine transforms, or random erasing (preserves text/logo details)

### Training Pipeline (Step-by-Step)

#### Step 1: Setup (Cell 0)
- Mount Google Drive
- Install required packages

#### Step 2: Configuration (Cell 1)
- Set project directory path
- Configure hyperparameters
- Define paths for training and test images

#### Step 3: Load and Process Data (Cell 3)
1. **Load CSVs**: Reads all brand-specific CSV files (excluding `test.csv`)
2. **Merge Data**: Combines all CSVs into a single DataFrame
3. **Build Image Paths**: Creates `image_path` column using pattern:
   ```
   {train_image_root}/{authentic|fake}/{brand_lowercase}/{Image Name}
   ```
4. **Train/Val Split**: 80-20 split stratified by Brand (ensures brand distribution in both sets)

#### Step 4: Data Preprocessing (Cells 5-6)
1. **Brand Encoder**: Fit `LabelEncoder` on training brands only
2. **Price Scaler**: Fit `StandardScaler` on log-transformed prices (log1p) from training set only
   - This ensures validation/test data are scaled using training statistics (prevents data leakage)

#### Step 5: Image Transforms (Cell 5)
- **Training**: Resize (224×224), RandomHorizontalFlip, RandomRotation(8°), ColorJitter, Normalize
- **Validation/Test**: Resize (224×224), Normalize only

#### Step 6: Dataset and DataLoader (Cell 7-8)
- Custom `SneakerDataset` class that:
  - Loads images, encodes brands, scales prices
  - Returns dictionary: `{'image': tensor, 'brand': tensor, 'price': tensor, 'label': tensor}`
- Creates train and validation DataLoaders

#### Step 7: Model Initialization (Cell 12)
- Creates `ViTMultiHeadFixed` model with pretrained ViT backbone
- Initializes optimizer (AdamW) and scheduler (CosineAnnealingLR)

#### Step 8: Training Loop (Cell 12)
For each epoch:
1. **Train**: Forward pass → compute combined loss → backward pass → optimizer step
2. **Validate**: Forward pass → compute accuracy
3. **Save Best Model**: Saves model with best validation accuracy
   - Saved checkpoint includes:
     - Model state dict
     - Brand encoder classes (for inference)
     - Price scaler statistics (mean, scale)

**Output**: Model saved to `best_model.pth` (or custom `SAVE_PATH`)

---

## Inference/Testing Process

### Step 1: Load Test Data (Cell 13-14)

1. **Load Test CSV**: Reads `test.csv` with flexible separator handling
2. **Initialize Test Dataset**: Uses `image_root_override` parameter to point to `data/sneakers/test/`
3. **Create Test DataLoader**: Batch size 32, no shuffling

### Step 2: Run Inference (Cell 17-18)

1. **Load Saved Model**: Load the best model checkpoint (if not already in memory)
2. **Predict**: For each batch:
   - Forward pass through model
   - Apply sigmoid to get probabilities
   - Threshold at 0.4 for binary predictions (reduces false positives)
3. **Compute Metrics** (if ground truth available):
   - Accuracy
   - AUC-ROC
4. **Save Results**: Appends `Pred_Prob` and `Pred_Label` columns to test DataFrame
   - Saves to `test_predictions.csv`

### Example Output

```csv
sln,Image Name,Brand,Price,Authentic,Pred_Prob,Pred_Label
1,test1.jpg,Adidas,45.00,0,0.23,0
2,test2.png,Nike,199.99,1,0.87,1
3,test3.jpg,Jordan,89.99,0,0.34,0
```

---

## How the Model Makes Decisions

### Factor 1: Image Analysis
- The ViT backbone extracts visual features from the sneaker image
- The model learns patterns that distinguish authentic vs. fake sneakers:
  - Logo quality and placement
  - Material texture and quality
  - Overall craftsmanship and detail
  - Brand-specific design elements

### Factor 2: Price Analysis
- Price is log-transformed and standardized using training statistics
- The model learns that:
  - Authentic sneakers typically have higher prices
  - Prices significantly below market value suggest counterfeits
- However, price alone is not sufficient (some authentic sales may be discounted)

### Factor 3: Brand Consistency
- The brand embedding helps the model:
  - Learn brand-specific authentic patterns
  - Ensure the image matches the stated brand
  - Use brand context to refine predictions

### Final Decision
The model combines all three factors:
```
Combined Features = [Image Embedding (512) | Brand Embedding (64) | Price Embedding (64)]
                    = 640-dimensional vector

Authenticity Score = AuthHead(Combined Features)
                    = Linear(640 → 512) → ReLU → Dropout(0.15)
                    → Linear(512 → 256) → ReLU → Dropout(0.25)
                    → Linear(256 → 1)

Probability = Sigmoid(Authenticity Score)  # Maps to [0, 1]

Prediction = {
    1 (Authentic) if Probability >= 0.4
    0 (Fake)      if Probability < 0.4
}
```

**Note**: The prediction threshold is set to 0.4 (instead of 0.5) to reduce false positives (authentic shoes incorrectly marked as fake).

---

## Expected Performance

Based on the latest training results:

### Training Results
- **Best Validation Accuracy**: 94.23%
- **Final Train Loss**: 0.108777
- **Final Validation Loss**: 0.435927

### Test Results
- **Test Accuracy**: 81.82%
- **Test AUC**: 0.8941

**Note**: Performance may vary based on dataset size and quality. The model uses partial fine-tuning (freezing first 2 layers) and minimal augmentation to preserve authentic details while preventing overfitting.

---

## Key Design Decisions

1. **Multimodal Fusion**: Combining image, brand, and price provides complementary signals
2. **Multi-Task Learning**: Brand classification as auxiliary task improves image representations and helps catch spelling mistakes in logos
3. **Contrastive Learning**: Helps cluster images by brand, improving generalization (weight: 0.03)
4. **Price Normalization**: Log-transform + standardization handles price skewness
5. **Stratified Split**: Ensures brand distribution in train/val sets
6. **Partial Fine-Tuning**: Freezes first 2 transformer layers (preserves general features) while fine-tuning remaining 10 layers (learns domain-specific patterns)
7. **Minimal Augmentation**: Preserves authentic details (text, logos) while still providing some regularization
8. **Lower Prediction Threshold**: 0.4 instead of 0.5 reduces false positives (authentic shoes incorrectly marked as fake)
9. **Early Stopping**: Prevents overfitting by stopping when validation accuracy doesn't improve for 7 epochs

---

## File Organization Summary

```
Project Root/
├── data/
│   └── sneakers/
│       ├── authentic/{brand}/{images}    # Training authentic images
│       ├── fake/{brand}/{images}         # Training fake images
│       └── test/{images}                 # Test images (flat)
├── Counterfeit_product_data - *.csv      # Training CSVs (one per brand)
├── test.csv                              # Test CSV
├── Model_evaluation_train_test.ipynb                   # Main notebook
└── best_model.pth                        # Saved model (after training)
```

---
