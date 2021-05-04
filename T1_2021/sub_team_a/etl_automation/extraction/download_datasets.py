from utils import helper
import os
import config as cfg



def main(output_dir):
	helper.create_dir(output_dir)
	for feat in cfg.FEAT_NCCOBS_CODES:
		print(feat)
		feat_code = cfg.FEAT_NCCOBS_CODES[feat]
		feat_pc_code = cfg.FEAT_PC_CODES[feat]
		download_url = cfg.BOM_DOWNLOAD_BASE_URL.format(cfg.STATION_CODE, feat_pc_code, feat_code)
		download_path = os.path.join(output_dir, '{}.zip'.format(feat))
		helper.download_zip(download_url, download_path)


if __name__=='__main__':
	main()

