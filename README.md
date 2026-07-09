# EECE 5644: Assignment #03: WheelsBazaar Used-Car Price Prediction

Predicts the **fair market price** of a used car from its attributes, for WheelsBazaar's
listing-review guardrails and the WheelsBazaar Direct instant-buy program.

## Files
| File | Description |
|------|-------------|
| `WheelsBazaar_Car_Price_Prediction.ipynb` | Main deliverable — fully executed notebook (EDA → cleaning → modelling → evaluation → business recommendations). |
| `WheelsBazaar_Car_Price_Prediction.html` | Rendered read-only version (open in any browser). |
| `wheelsbazaar_price_model.joblib` | Trained scikit-learn pipeline (preprocessing + best model), ready to `joblib.load` and `.predict`. |
| `build_notebook.py` | Script that generates the notebook (reproducibility). |
| `car details v4.csv` | Dataset (2,059 listings, 20 attributes). |

## Approach
- **Target:** model `log(Price)` (raw skew 4.97 → 0.49) so the model optimises relative error, then report errors in rupees.
- **Feature engineering:** parse `Engine`/`Max Power`/`Max Torque` strings to numbers, derive `Age`, cap odometer outliers, drop high-cardinality `Model`.
- **Leakage-safe pipeline:** `ColumnTransformer` (median-impute + scale numeric; most-frequent-impute + one-hot categorical) inside a `Pipeline`; 80/20 train/test split + 5-fold CV.
- **Models compared:** Ridge (baseline), Random Forest, Histogram Gradient Boosting.

## Result
**Histogram Gradient Boosting** was selected (lowest typical-car error):

| Metric | Value |
|--------|-------|
| CV R² (log-price) | 0.94 |
| MAE | ₹278,760 |
| MAPE | 12.6% |
| Within ±20% of actual | 81.8% |

Top price drivers (permutation importance): car age, engine power, brand, and kilometers driven.

## Reproduce
```bash
python3 build_notebook.py
python3 -m jupyter nbconvert --to notebook --execute --inplace WheelsBazaar_Car_Price_Prediction.ipynb
```

## Load the trained model
```python
import joblib, pandas as pd
model = joblib.load("wheelsbazaar_price_model.joblib")
price = np.expm1(model.predict(new_listings_df))   # model returns log-price
```
