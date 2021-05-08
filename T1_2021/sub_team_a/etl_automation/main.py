from extraction import download_datasets
from transformation import merge_data_sources, clean_data
import config as cfg


if __name__ == '__main__':
	# download_datasets.main(cfg)
	merge_data_sources.main(cfg)
	clean_data.main(cfg)