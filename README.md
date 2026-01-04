# Satellite-Based Property Valuation

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning project that combines satellite imagery with traditional housing features to predict property values. This research explores whether visual neighborhood context can enhance real estate valuation models.

**Table of Contents**
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Models](#models)
- [Results](#results)
- [Setup](#setup)
- [Usage](#usage)

---

## Overview

This project develops a comprehensive property valuation system that integrates:

**Traditional Features:**
- Property size (living area and lot size)
- Structural details (bedrooms, bathrooms)
- Quality metrics and construction year
- Location coordinates

**Visual Context:**
- Neighborhood density patterns
- Green space distribution  
- Water body proximity
- Transportation infrastructure

The primary objective is to determine whether satellite imagery provides meaningful improvements to property valuation accuracy when combined with structured data.

## Problem Statement

Traditional property valuation models rely on structured attributes such as square footage, number of rooms, and basic location data. However, these models often miss important neighborhood characteristics that significantly impact property values.

**Current Limitations:**
- Limited neighborhood context
- Missing environmental factors
- Lack of visual spatial information

**Potential Satellite Image Benefits:**
- Water bodies and natural features
- Green space vs urban density
- Road networks and accessibility
- Overall neighborhood development patterns

**Research Question:** Can satellite imagery enhance property valuation accuracy when combined with traditional tabular features?

## Methodology

The project implements a multimodal machine learning pipeline:

```
Property Location (Lat/Lng)
           |
    ┌──────┴──────┐
    |             |
    v             v
Satellite API  Tabular Data
    |          (sqft, beds, etc.)
    v             |
CNN Features      |
(ResNet18)        |
    |             |
    └──────┬──────┘
           |
           v
    Fusion Models
           |
           v
   Price Prediction
```

**Pipeline Steps:**
1. Data preprocessing and cleaning
2. Satellite image acquisition via API
3. Feature extraction using pre-trained CNN
4. Model training (tabular, image, fusion)
5. Performance evaluation and comparison

## Project Structure

```
satellite-property-valuation/
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned data
│   └── images/                 # Satellite images
├── notebooks/
│   ├── 01_preprocessing.ipynb        # Data preparation
│   ├── 02_tabular_model.ipynb        # Baseline model  
│   ├── 03_image_model.ipynb          # CNN model
│   ├── 04_fusion_model.ipynb         # Combined approach
│   ├── 05_grad_cam.ipynb             # Model interpretation
│   └── 06_evaluation.ipynb           # Results analysis
├── src/
│   └── data_fetcher.py         # Image acquisition
├── outputs/
│   └── predictions.csv         # Final results
├── requirements.txt
└── README.md
```

## Models

Three distinct approaches were implemented and evaluated:

### 1. Tabular Baseline Model
- Uses only traditional property features
- Implements various regression algorithms
- Serves as performance benchmark
- **Status: Best performing model**

### 2. Image-Only Model  
- ResNet18 CNN for feature extraction
- Satellite images as sole input
- Tests pure visual prediction capability
- **Status: Limited predictive power**

### 3. Multimodal Fusion Model
- Early fusion of tabular and image features
- Combines CNN embeddings with structured data
- Explores complementary information integration
- **Status: No improvement over baseline**

## Results

| Model Type | RMSE | R² Score | Performance |
|------------|------|----------|-------------|
| Tabular Only | Best | Highest | Winner |
| Image Only | Poor | Negative | Insufficient |
| Multimodal Fusion | Worse | Lower | No improvement |

### Key Findings

**Tabular Features Dominate:** Traditional property characteristics (size, location, quality) provide the strongest predictive signal for property valuation.

**Satellite Images Show Promise:** While not improving overall accuracy, Grad-CAM analysis reveals that the CNN learns meaningful spatial patterns including water bodies, green spaces, and urban density.

**Fusion Challenges:** Simple concatenation of high-dimensional image features with tabular data introduces noise rather than complementary information.

### Model Interpretation

Grad-CAM visualizations show the CNN focuses on:
- **High-value properties:** Water access, green areas, open layouts
- **Lower-value properties:** Dense development, industrial areas, limited amenities

This validates that satellite imagery captures relevant neighborhood context, even when it doesn't improve prediction accuracy.

## Setup

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/satellite-property-valuation.git
cd satellite-property-valuation
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API access** (Optional)
Create a `.env` file for satellite image downloading:
```env
MAPBOX_TOKEN=your_mapbox_api_key_here
```

### Download Satellite Images (Optional)
```bash
python src/data_fetcher.py
```
*Note: Images are not included in the repository due to size and API constraints.*

## Usage

### Execution Order

Run the notebooks in sequence for complete analysis:

| Notebook | Description |
|----------|-------------|
| `01_preprocessing.ipynb` | Data cleaning and exploratory analysis |
| `02_tabular_model.ipynb` | Baseline model development |
| `03_image_model.ipynb` | CNN model for satellite images |
| `04_fusion_model.ipynb` | Multimodal approach |
| `05_grad_cam.ipynb` | Model interpretation and visualization |
| `06_evaluation.ipynb` | Comparative analysis and results |

### Generating Predictions

Final test predictions are saved to:
```
outputs/predictions.csv
```

Format:
```csv
id,predicted_price
1,285000
2,342000
```

### Notes and Limitations

- Satellite imagery serves as complementary data, not a primary signal
- Current fusion approach uses simple concatenation; more sophisticated methods may yield better results
- Image acquisition requires API access and may have rate limits
- Results emphasize model interpretability over raw performance metrics
