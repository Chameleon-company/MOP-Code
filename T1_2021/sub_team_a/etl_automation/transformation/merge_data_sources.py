import os
from utils import helper


def main(output_dir, cfg):
	extraction_dir = os.path.join(output_dir, cfg.EXTRACTION_DIR) 
	output_dir = os.path.join(output_dir, cfg.TRANSFORMATION_DIR)
	helper.create_dir(output_dir)
	features_file_list = helper.get_file_list_from_dir(extraction_dir, 'csv')
	if len(features_file_list)>1:
		base_df = helper.read_csv_file(features_file_list[0])
		base_df = helper.convert_to_datetime(base_df, list(cfg.FEAT_COL_NAME.values()),cfg.DATE_COLUMN)
		for feat_file in features_file_list[1:]:
			print(feat_file)
			feat_df = helper.read_csv_file(feat_file)
			feat_df = helper.convert_to_datetime(feat_df, list(cfg.FEAT_COL_NAME.values()),cfg.DATE_COLUMN)
			base_df = helper.merge_files(base_df, feat_df, merge_type='inner', left_on=cfg.DATE_COLUMN, right_on=cfg.DATE_COLUMN)

	base_df = base_df[base_df[cfg.DATE_COLUMN]> cfg.BOM_CUTOFF_DATE]
	base_df.to_csv(os.path.join(output_dir, cfg.MERGED_FEATURES_FILE), index=False)