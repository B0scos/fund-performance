from src.process.load_raw import ProcessRaw
from src.process.clean_data import DataCleaner, DataCleanerConfig

pr = ProcessRaw()

df = pr.concat()
try:
    pr.save(df, filename="interim.parquet", fmt="parquet", target="interim")
except Exception as e:
    print(f"Warning: failed to save interim data: {e}")

# Run data cleaning (skeleton; no destructive logic by default)
cfg = DataCleanerConfig(required_columns=["fund_cnpj", "report_date"])
dc = DataCleaner(config=cfg)
cleaned = dc.run(df, save=True, filename="cleaned.parquet", fmt="parquet")