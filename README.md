# Breast Cancer Classification (EDA + Preprocessing)

This project explores the Wisconsin Breast Cancer dataset to understand feature distributions, check data quality, reduce multicollinearity, and prepare the data for modeling.

## Files
- `breast-cancer.ipynb`: Main notebook with EDA and preprocessing steps.
- `breast-cancer.csv`: Dataset used in the notebook.
- `README.md`: Project overview and usage instructions.

## Environment and Dependencies
Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn missingno
```

## Notebook Workflow
1. Load data and inspect structure (`df.info()`, `df.describe()`).
2. Verify missing values using `missingno` and `df.isnull().sum()`.
3. Initial class balance visualization with `seaborn.countplot`.
4. Encode target: map `diagnosis` from `{'M': 1, 'B': 0}`.
5. Distribution plots (KDE) for all features to assess skewness and scale.
6. Correlation heatmap to detect multicollinearity.
7. Drop identifier column `id`.
8. Remove highly correlated features (absolute correlation > 0.92) to reduce redundancy.

## Visualizations
- Missingness overview using `missingno.bar` to confirm completeness of all columns.
- Class distribution with `seaborn.countplot` of `diagnosis`.
- Kernel Density Estimation (KDE) plots of all features to assess distribution shapes and scale.
- Correlation heatmap (upper triangle masked) to visualize multicollinearity patterns across features.

## Key Findings
- No missing values detected in any column.
- `diagnosis` encoded to binary: Malignant → 1, Benign → 0.
- Many feature distributions are right-skewed; `se` features show small variance.
- Multicollinearity present; after dropping `id` and highly correlated features, columns reduced to 23.

## Inferences from the Notebook
- The dataset is clean and ready for modeling without imputation.
- Feature distributions indicate substantial right skew; scaling and possibly transformations may help many models.
- Strong correlation groups exist among size-related features (e.g., radius, perimeter, area and their worst/mean variants), suggesting redundancy.
- Removing features with absolute correlation > 0.92 reduces dimensionality from 32 to 23 columns, which should help mitigate overfitting and improve model stability.
- The class count plot shows both malignant and benign classes are present; evaluation should consider metrics beyond accuracy (e.g., recall for malignant cases).

## How to Run
1. Ensure `breast-cancer.csv` is in the project root.
2. Open and run `breast-cancer.ipynb` in Jupyter or VS Code.

## Notes
- You may see a seaborn deprecation warning about `palette` without `hue` on newer versions; it's safe to ignore or set `hue='diagnosis'` with `legend=False` for the same visual effect.

## Next Steps
- Scale features and train baseline classifiers (e.g., Logistic Regression, SVM).
- Perform cross-validation and compare metrics (accuracy, precision, recall, ROC-AUC).
