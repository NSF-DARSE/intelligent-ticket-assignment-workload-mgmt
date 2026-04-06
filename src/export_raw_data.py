import psycopg2
import pandas as pd

# Database connection
conn = psycopg2.connect(
    host="localhost",
    port="5432",
    database="Intelligent Ticket Assignment & Workload Management",
    user="postgres",
    password="2014"
)

# Read table into pandas
query = "SELECT * FROM autotask_raw;"
df = pd.read_sql(query, conn)

# Save to Raw_Data folder
file_path = "data/Raw_Data/autotask_raw_data.csv"
df.to_csv(file_path, index=False)

print("Raw data exported successfully!")
print("File saved at:", file_path)

