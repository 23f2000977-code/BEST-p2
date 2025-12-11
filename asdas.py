import pandas as pd
import json
from io import StringIO

# The data from your logs
csv_data = """ID,Name,Joined,Value
3,Beta,02/01/24,10
1,Alpha,2024-01-30,5
2,Gamma,1 Feb 2024,7"""

# Read and Normalize
df = pd.read_csv(StringIO(csv_data))

# 1. Lowercase columns
df.columns = df.columns.str.lower().str.strip()

# 2. Convert Dates to YYYY-MM-DD (No Time component!)
df['joined'] = pd.to_datetime(df['joined'], dayfirst=True).dt.strftime('%Y-%m-%d')

# 3. Sort by ID
df = df.sort_values('id')

# 4. Convert to list of dicts
data = df.to_dict(orient='records')

# 5. Print as JSON string
print(json.dumps(data, separators=(',', ':')))