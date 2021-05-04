from utils import helper
import os



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


def main(output_dir, cfg):
	output_dir = os.path.join(output_dir, cfg.EXTRACTION_DIR)
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

if __name__=='__main__':
	main()

