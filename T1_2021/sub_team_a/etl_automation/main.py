from extraction import download_datasets
from transformation import merge_data_sources
import config as cfg


if __name__ == '__main__':
	# download_datasets.main(cfg.OUTPUT_DIR, cfg)
	merge_data_sources.main(cfg.OUTPUT_DIR, cfg)