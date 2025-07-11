# ------------- DATA FOUNDATION (Stage 1) -----------------
make ingest          # loads CSVs, writes 01_data_quality_report.json
make augment         # outputs 02_augmented_calls.csv
make econ            # downloads econ indicators → 03_econ_features.csv

# ------------- ADVANCED EDA (Stage 2) --------------------
make eda_ts          # decomposition, lag heat-map, seasonality plot
make mail_eff        # campaign effectiveness bar-chart + CSV
make econ_sens       # call-vs-econ heat-map
make patterns        # anomalies + Granger causality

# ------------- MODELLING (Stage 3) -----------------------
make features        # builds 50 + engineered features
make model_train     # SARIMAX, Prophet, XGBoost CV metrics
make interpret       # SHAP summary + top-5 PDPs

# ------------- DELIVERY (Stage 4) ------------------------
make dashboard       # launches Dash on http://127.0.0.1:8050
make ppt            # generates output/COO_presentation.pptx




make ingest augment econ eda_ts mail_eff econ_sens patterns features model_train interpret ppt
