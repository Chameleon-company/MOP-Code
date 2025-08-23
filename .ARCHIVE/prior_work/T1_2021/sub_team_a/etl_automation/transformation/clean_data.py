from utils import helper
import numpy as np
import os


# update the missing values using up and down lookup mean 
def perform_imputation(df, col_name):
	def update_missing_value(row_idx):
		nonlocal df, col_name
		row_lookup = 4      # lookup above and below the missing value
		missing_val_lookup_mean = df.iloc[row_idx-row_lookup:row_idx+row_lookup].fillna(value=0)[col_name].tolist()
		return np.mean(missing_val_lookup_mean)
	missing_col_rows = df[df[col_name].isnull()]
	for missing_idx in missing_col_rows.index:
		df.at[missing_idx, col_name] = update_missing_value(missing_idx)
	return df



def main(cfg):
	output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.TRANSFORMATION_DIR)
	prev_stage_output = os.path.join(output_dir, cfg.MERGED_FEATURES_FILE)
	pred_stage_df = helper.read_csv_file(prev_stage_output, delim='\t')
	# columns that contains missing value
	columns_with_nan_values = pred_stage_df.columns[pred_stage_df.isna().any()].tolist()      
	for missing_val_col in columns_with_nan_values:
		pred_stage_df = perform_imputation(pred_stage_df, missing_val_col)
	pred_stage_df.to_csv(os.path.join(output_dir, cfg.CLEANED_FEATURES_FILE), index=False, sep='\t')