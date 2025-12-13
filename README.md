# Sneaker Authentic vs Fake Classification

A deep learning system for detecting counterfeit branded sneakers using a **multimodal Vision Transformer (ViT)** approach that combines image analysis, brand recognition, and price information to determine authenticity.

## Project Overview

This project implements a binary classifier that predicts whether a sneaker listing is **Authentic (1)** or **Fake (0)**. The model leverages:

1. **Image Features**: Vision Transformer (ViT-Base) extracts visual features from sneaker images, learning to recognize authentic vs. fake characteristics
2. **Brand Information**: Embedding layer maps brand names to learned representations, helping the model understand brand-specific authentic patterns
3. **Price Information**: Price is used as a decision factor (generally, authentic sneakers have higher prices than counterfeits)

### Model Architecture

The model (`ViTMultiHeadFixed`) consists of:

1. **Image Backbone**: Pretrained ViT-Base-Patch16-224 from `timm` library
   - Extracts 768-dimensional image features
   - Projected to 512-dimensional embeddings

2. **Brand Embedding**: Learnable embedding layer (64 dimensions) for brand classification
   - Maps brand names (Adidas, Nike, Jordan, Puma, Reebok) to dense vectors

3. **Price Projection**: MLP that projects price (log-normalized and standardized) to 64 dimensions

4. **Multi-Head Output**:
   - **Authenticity Head**: Combines image (512) + brand (64) + price (64) → predicts Authentic/Fake
     - Architecture: 512 → 128 → 1 (with ReLU and Dropout)
   - **Brand Classification Head**: Uses image features alone → predicts brand (auxiliary task)
     - Architecture: 512 → 256 → num_brands (with ReLU and Dropout)
   - **Contrastive Learning Branch**: Projects image embeddings for supervised contrastive loss

### Training Strategy

The model is trained with three loss components:

1. **Primary Loss**: Binary Cross-Entropy (BCE) for authenticity prediction
2. **Auxiliary Loss**: Cross-Entropy for brand classification (weight: 0.5)
3. **Contrastive Loss**: Supervised contrastive loss for brand-based image clustering (weight: 0.1)

This multi-task learning approach helps the model learn better image representations by:
- Forcing the model to identify brands from images
- Encouraging similar images of the same brand to cluster together
- Using price as additional signal for authenticity

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

Key hyperparameters (editable in Cell 1):

```python
BATCH_SIZE = 24
IMG_SIZE = 224
NUM_EPOCHS = 12
LR = 2e-4
WEIGHT_DECAY = 1e-4
BACKBONE = 'vit_base_patch16_224'
FREEZE_BACKBONE = False
BRAND_EMBED_DIM = 64
OUT_DIM = 512
USE_CONTRASTIVE = True
AUX_BRAND_LOSS = True
```

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

### Step 2: Run Inference (Cell 15-16)

1. **Load Saved Model**: Load the best model checkpoint (if not already in memory)
2. **Predict**: For each batch:
   - Forward pass through model
   - Apply sigmoid to get probabilities
   - Threshold at 0.5 for binary predictions
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
Authenticity Score = AuthHead(Combined Features)
Prediction = 1 if sigmoid(Authenticity Score) >= 0.5 else 0
```

---

## Expected Performance

Based on the training notebook:
- **Validation Accuracy**: ~75%+ (after epoch 1)
- **Test Accuracy**: ~84.62%
- **Test AUC**: ~0.95

Note: Performance may vary based on dataset size and quality.

---

## Key Design Decisions

1. **Multimodal Fusion**: Combining image, brand, and price provides complementary signals
2. **Multi-Task Learning**: Brand classification as auxiliary task improves image representations
3. **Contrastive Learning**: Helps cluster images by brand, improving generalization
4. **Price Normalization**: Log-transform + standardization handles price skewness
5. **Stratified Split**: Ensures brand distribution in train/val sets
6. **Frozen vs. Fine-tuned Backbone**: ViT backbone is fine-tuned (not frozen) for domain adaptation

---

## Troubleshooting

### Common Issues

1. **Missing Images**: If images are not found, check:
   - Image filenames match exactly (case-sensitive)
   - Brand folder names are lowercase
   - Path structure matches expected pattern

2. **CSV Column Mismatch**: Ensure all required columns are present:
   - `Image Name`, `Brand`, `Price`, `Authentic`

3. **Price Scaling Error**: Make sure price scaler is fit on training data only (done automatically)

4. **Brand Encoding Error**: Ensure test brands are in the training brand set, or handle unknown brands in preprocessing

5. **CUDA Out of Memory**: Reduce `BATCH_SIZE` or `IMG_SIZE`

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
├── training3 (1).ipynb                   # Main notebook
└── best_model.pth                        # Saved model (after training)
```

---