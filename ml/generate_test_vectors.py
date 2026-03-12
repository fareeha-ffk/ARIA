import pandas as pd

ML_DIR = '/Users/fareeha/ARIA/ml/'
RTL_DIR = '/Users/fareeha/ARIA/rtl/'

df = pd.read_csv(ML_DIR + 'golden_vectors.csv')

# Take first 100 vectors
df = df.head(100)

# Write as Verilog memory file
with open(RTL_DIR + 'test_vectors.mem', 'w') as f:
    for _, row in df.iterrows():
        # Write 5 INT8 inputs as hex bytes
        for col in ['in_q0','in_q1','in_q2','in_q3','in_q4']:
            val = int(row[col]) & 0xFF  # convert to unsigned byte
            f.write(f'{val:02x}\n')
        # Write expected label
        f.write(f'{int(row["label"]):02x}\n')

print(f"Generated {len(df)} test vectors")
print(f"Saved to {RTL_DIR}test_vectors.mem")
# Updated for Week 1 & 2
