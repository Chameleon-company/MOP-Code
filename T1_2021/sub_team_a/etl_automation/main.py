from extraction import download_datasets
import config as cfg


if __name__ == '__main__':
	download_datasets.main(cfg.OUTPUT_DIR, cfg)