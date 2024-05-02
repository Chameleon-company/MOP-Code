from extraction import download_datasets
from transformation import merge_data_sources, clean_data
from model import time_series
from load import load_data_to_server
import config as cfg


if __name__ == '__main__':
	download_datasets.main(cfg)
	merge_data_sources.main(cfg)
	clean_data.main(cfg)
	time_series.main(cfg)
	load_data_to_server.copy_ftp_file_over_sftp_using_key(cfg)