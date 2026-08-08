import pandas as pd
import os

# 1. Test Data Prep script logic locally
raw_df = pd.read_csv('used_cars.csv')
print(f"Raw Data Shape: {raw_df.shape}")

# 2. Check if prep.py outputs train.csv and test.csv when run
os.system("python data-science/src/prep.py --raw_data used_cars.csv --test_train_ratio 0.2 --train_data ./output/train --test_data ./output/test")

# 3. Verify created output files
train_exists = os.path.exists("./output/train/train.csv")
test_exists = os.path.exists("./output/test/test.csv")

print(f"Train CSV Generated: {train_exists}")
print(f"Test CSV Generated: {test_exists}")
