from langchain.pipeline import MultimodalPipeline
import pandas as pd

# Step 1: Load Data
data = pd.read_csv("C:/Users/nairs5/OneDrive - UT Arlington/Research/Datathon/us_congestion_2016_2022_sample_2m.csv")

# Step 2: Extract Textual Data
# Assuming your data contains columns with textual descriptions
textual_data = data["Description"]

# Step 3: Initialize Langchain Pipeline
pipeline = MultimodalPipeline()

# Step 4: Analyze Textual Data with LLAMA
llama_results = pipeline.llama_analysis(textual_data)

# Step 5: Print Insights
print("LLAMA Insights:")
print(llama_results)
