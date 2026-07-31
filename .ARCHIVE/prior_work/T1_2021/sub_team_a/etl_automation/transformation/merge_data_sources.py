import os
from utils import helper


def merge_holiday_data(df, cfg):
	holiday_df = helper.read_csv_file(cfg.HOLIDAY_FILE_PATH, '\t')
	holiday_df = helper.get_datetime_column(holiday_df, cfg.DATE_COLUMN)
	df = helper.merge_files(df, holiday_df, merge_type='left', left_on=cfg.DATE_COLUMN, right_on=cfg.DATE_COLUMN)
	df[cfg.HOLIDAY_COLUMN_NAME] = df[cfg.HOLIDAY_COLUMN_NAME].fillna(value=0).astype(str)
	return df


def merge_restriction_data(df, cfg):
	restriction_df = helper.read_csv_file(cfg.COVID_RESTRICTION_FILE_PATH, '\t')
	restriction_df = helper.get_datetime_column(restriction_df, cfg.DATE_COLUMN)
	df = helper.merge_files(df, restriction_df, merge_type='left', left_on=cfg.DATE_COLUMN, right_on=cfg.DATE_COLUMN)
	df[cfg.COVID_RESTRICTION_COLUMN_NAME] = df[cfg.COVID_RESTRICTION_COLUMN_NAME].fillna(value=0).astype(str)
	return df
	

def merge_pedestrian_count(extraction_dir, df, cfg):
	ped_df = helper.read_csv_file(os.path.join(extraction_dir, cfg.MP_DOWNLOAD_FILE), '\t')
	ped_df = helper.get_datetime_column(ped_df, cfg.DATE_COLUMN)
	df = helper.merge_files(df, ped_df, merge_type='inner', left_on=cfg.DATE_COLUMN, right_on=cfg.DATE_COLUMN)
	return df


def main(cfg):
	output_dir = cfg.OUTPUT_DIR
	extraction_dir = os.path.join(output_dir, cfg.EXTRACTION_DIR) 
	output_dir = os.path.join(output_dir, cfg.TRANSFORMATION_DIR)
	helper.create_dir(output_dir)
	# Merging feature data
	features_file_list = helper.get_file_list_from_dir(extraction_dir, 'csv')
	if len(features_file_list)>1:
		# Merge Feature flat files
		base_df = helper.read_csv_file(features_file_list[0])
		base_df = helper.convert_to_datetime(base_df, list(cfg.FEAT_COL_NAME.values()), cfg.DATE_COLUMN)
		for feat_file in features_file_list[1:]:
			if cfg.MP_DOWNLOAD_FILE not in feat_file:
				print(feat_file)
				feat_df = helper.read_csv_file(feat_file)
				feat_df = helper.convert_to_datetime(feat_df, list(cfg.FEAT_COL_NAME.values()),cfg.DATE_COLUMN)
				base_df = helper.merge_files(base_df, feat_df, merge_type='inner', left_on=cfg.DATE_COLUMN, right_on=cfg.DATE_COLUMN)

	# dataset date cutoff point
	base_df = base_df[base_df[cfg.DATE_COLUMN] >= cfg.BOM_CUTOFF_DATE]
	base_df = merge_holiday_data(base_df, cfg)
	base_df = merge_restriction_data(base_df, cfg)
	base_df = merge_pedestrian_count(extraction_dir, base_df, cfg)
	base_df.to_csv(os.path.join(output_dir, cfg.MERGED_FEATURES_FILE), index=False, sep='\t')