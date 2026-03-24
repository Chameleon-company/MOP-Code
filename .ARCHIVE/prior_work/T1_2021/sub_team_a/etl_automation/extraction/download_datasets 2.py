from utils import helper
import os
import sodapy


def create_download_url(cfg, feat):
	feat_code = cfg.FEAT_NCCOBS_CODES[feat]
	feat_pc_code = cfg.FEAT_PC_CODES[feat]
	download_url = cfg.BOM_DOWNLOAD_BASE_URL.format(cfg.STATION_CODE, feat_pc_code, feat_code)
	return download_url


def download_zip_file(url, path):
	helper.download_file(url, path)


def unzip_zip_file(zip_file_path, unzip_dir):
	helper.extract_zip_dir(zip_file_path, unzip_dir)


def get_csv_file_from_dir(dir_path):
	return helper.get_file_list_from_dir(dir_path, '.csv')[0]


def get_pedestrian_count(output_dir, cfg):
	client = Socrata(cfg.MP_URI_ENDPOINT, None)
	results = client.get('b2ak-trbp', limit=3482938)
	ped_df = helper.get_dataframe_from_json(results)
	# for testing
	# ped_df = helper.read_csv_file('test.tsv', delim='\t')
	ped_df[cfg.DATE_COLUMN] = ped_df['year'].astype(str)  + '-' + ped_df['month']  + '-' + ped_df['mdate'].astype(str) 
	ped_df = helper.get_datetime_column(ped_df, cfg.DATE_COLUMN)
	# converting hourly data to daily count
	ped_df['hourly_counts'] = ped_df['hourly_counts'].astype('int')
	ped_df = ped_df.groupby([cfg.DATE_COLUMN], as_index=False)['hourly_counts'].sum()
	ped_df.rename(columns={'hourly_counts':'Total_Pedestrian_Count'}, inplace=True)
	ped_df.to_csv(os.path.join(output_dir, cfg.MP_DOWNLOAD_FILE), index=False, sep='\t')


def main(cfg):
	# downloading feature data
	output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.EXTRACTION_DIR)
	for feat in cfg.FEAT_NCCOBS_CODES:
		print(feat)
		feat_dir_path = os.path.join(output_dir, feat)
		helper.create_dir(feat_dir_path)
		feat_zip_download_path = os.path.join(feat_dir_path, '{}.zip'.format(feat))
		feat_download_url = create_download_url(cfg, feat)
		download_zip_file(feat_download_url, feat_zip_download_path)
		unzip_zip_file(feat_zip_download_path, feat_dir_path)
		feat_csv_file_path = get_csv_file_from_dir(feat_dir_path)
		helper.move_file(feat_csv_file_path, os.path.join(output_dir, '{}.csv'.format(feat)))
		helper.remove_dir(feat_dir_path)

	get_pedestrian_count(output_dir, cfg)

if __name__=='__main__':
	main()