import pandas as pd

# Load the two CSV files
query_moment_pair_df = pd.read_csv('./data/TVR_Ranking_old/raw_query_moment_pair.csv')
annotation_df = pd.read_csv('./data/TVR_Ranking_old/raw_annotation.csv')

# Merge the two dataframes on 'pair_id'
merged_df = pd.merge(query_moment_pair_df, annotation_df, on='pair_id', how='inner')

# Optionally, you might want to reorder the columns or make other adjustments here

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('./data/TVR_Ranking/raw_query_moment_pair_annotation.csv', index=False)

print("The merged file has been created successfully.")