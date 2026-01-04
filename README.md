# House-Price-Prediction
This multimodal pipeline predicts house prices by fusing tabular data with visual context from satellite imagery. Using CNNs for feature extraction, it benchmarks a Model Zoo of regressors. Grad-CAM heatmaps provide explainability for visual property value drivers like density and water proximity.

This README provides a comprehensive guide to setting up and executing the **Multimodal Satellite Imagery and Property Valuation Pipeline**, based on the provided development sources.

# Project Overview: Satellite Imagery-Based Property Valuation

This project aims to enhance traditional real estate valuation by developing a **Multimodal Regression Pipeline**. By combining historical tabular housing data with programmatically acquired satellite imagery, the model integrates "curb appeal" and environmental context—such as green cover and neighborhood density—to predict property market values more accurately.

---

# Environment Setup

### 1. Prerequisites
Ensure you have Python 3.10+ installed. It is recommended to use a virtual environment (Conda or venv) to manage dependencies.

### 2. Dependencies
Based on the project's research and multimodal requirements, install the following libraries:

```bash
# Core Data Handling & Visualization
pip install pandas numpy matplotlib seaborn geopandas pillow requests

# Machine Learning & Optimization
pip install scikit-learn xgboost lightgbm joblib

# Deep Learning & Image Processing
pip install torch torchvision opencv-python
pip install tensorflow==2.10.1 keras efficientnet
pip install patchify segmentation-models protobuf==3.20.3
```

---

# System Architecture

The project utilizes a **Multimodal Fusion Architecture** to process disparate data types simultaneously:

1.  **Tabular Branch:** Processes numerical and categorical features (e.g., bedrooms, bathrooms, `sqft_living`). Features undergo log-transformation and encoding.
2.  **Image Branch:** Uses a pre-trained Convolutional Neural Network (CNN), such as **ResNet18** or **EfficientNet B2**, as a feature extractor to convert satellite images into high-dimensional visual embeddings.
3.  **Fusion Layer:** Concatenates the visual embeddings with the processed tabular features into a single feature vector.
4.  **Regressor (Model Zoo):** The combined data is fed into various regression models (XGBoost, Random Forest, LightGBM) to predict the final property price.
5.  **Explainability:** Employs **Grad-CAM** to visually highlight which regions of the satellite imagery (e.g., water proximity or foliage) most influenced the prediction.

---

# Complete Workflow

### Step 1: Data Acquisition
Run `data_fetcher.py` to programmatically acquire satellite images. The script uses the `lat` and `long` coordinates from the base dataset to fetch images via APIs such as ArcGIS, Google Maps Static, or Mapbox.

### Step 2: Exploratory Data Analysis (EDA) & Preprocessing
Use `preprocessing.ipynb` to clean and prepare the tabular data. 
*   **Outlier Removal:** Handle anomalies, such as correcting houses listed with 33 bedrooms.
*   **Feature Engineering:** Calculate new metrics like `basement_ratio` and `living_vs_neighbors`.
*   **Transformation:** Apply Log-transformation to skewed features like `price`, `sqft_lot`, and `sqft_living` to improve model convergence.

### Step 3: Feature Extraction & Fusion
In the modeling phase, satellite images are resized to 224x224 and passed through the CNN backbone.
*   Extract 512-dimensional embeddings (for ResNet18) or use PCA to reduce dimensions (for EfficientNet).
*   Merge these embeddings with the tabular data frame.

### Step 4: Model Training (Model Zoo)
Run `model_training.ipynb` to evaluate the **Model Zoo**. The pipeline systematically compares:
*   Linear Models: Ridge and Lasso Regression.
*   Ensemble Models: Random Forest and Gradient Boosting.
*   Boosting Algorithms: XGBoost and LightGBM.

### Step 5: Evaluation & Explainability
*   **Metrics:** Models are evaluated based on **RMSE** and **R² Score**.
*   **Visualization:** Generate Grad-CAM heatmaps to interpret the visual features driving property value.

---

# Project Structure & Deliverables

To ensure a valid submission, the repository must contain the following files:

*   `data_fetcher.py`: Script for API image downloads.
*   `preprocessing.ipynb`: Notebook for data cleaning and transformation.
*   `model_training.ipynb`: The multimodal training loop and evaluation.
*   `enrollno_final.csv`: Final predictions (format: `id, predicted_price`).
*   `enrollno_report.pdf`: Documentation of EDA, architecture, and visual insights.

---

# Conclusion

By integrating spatial visual data with traditional metrics, this pipeline captures the "environmental context" often missed by standard models. The systematic testing of the Model Zoo ensures that the best-performing algorithm is selected, while Grad-CAM provides the necessary transparency for real estate stakeholders to understand the impact of visual characteristics on property valuation.
