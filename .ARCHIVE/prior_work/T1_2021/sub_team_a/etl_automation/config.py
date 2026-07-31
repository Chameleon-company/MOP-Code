### MAIN EXECUTION ##################################################
OUTPUT_DIR = 'op'
#####################################################################

### Extraction ######################################################

# Olympia PARK
STATION_CODE = '086338'

FEAT_NCCOBS_CODES = {
	'rainfall': 136,
	'max_temp' : 122,
	'min_temp' : 123,
	'solar_exposure' : 193

}

FEAT_PC_CODES = { 
	'rainfall': -1490877699,
	'max_temp' : -1490874810,
	'min_temp' : -1490875006,
	'solar_exposure' : -1490892702

}

# BOM DOWNLOAD URL PATTERN
BOM_DOWNLOAD_BASE_URL = 'http://www.bom.gov.au/jsp/ncc/cdio/weatherData/av?p_display_type=dailyZippedDataFile&p_stn_num={}&p_c={}&p_nccObsCode={}&p_startYear=2021'

EXTRACTION_DIR = 'extraction'

# Melbourne Portal Resource
MP_URI_ENDPOINT = 'data.melbourne.vic.gov.au'
MP_DOWNLOAD_FILE = 'pedestrian.tsv'
#####################################################################



### Tranform ########################################################

TRANSFORMATION_DIR = 'transform'

FEAT_COL_NAME = {
	'rainfall': 'Rainfall amount (millimetres)',
	'max_temp' : 'Maximum temperature (Degree C)',
	'min_temp' : 'Minimum temperature (Degree C)',
	'solar_exposure' : 'Daily global solar exposure (MJ/m*m)'
}

DATE_COLUMN = 'Date'


# USE BOM DATA AFTER THIS DATE
BOM_CUTOFF_DATE = '2015-01-01'



HOLIDAY_FILE_PATH = 'user_input/holidays.tsv'
HOLIDAY_COLUMN_NAME = 'Public_Holiday'


COVID_RESTRICTION_FILE_PATH = 'user_input/restriction.tsv'
COVID_RESTRICTION_COLUMN_NAME = 'Covid Restrictions'

# STAGES OUTPUT FILES IN CHRONOLOGICAL ORDER
MERGED_FEATURES_FILE = 'features_merged.tsv'
CLEANED_FEATURES_FILE = 'clean_features.tsv'
#####################################################################



### Model ###########################################################

MODEL_DIR = 'model'
FE_PAST_N_DAYS = 350
# Because the data is 2 weeks late, we predict the next 3 weeks.
PREDICT_NEXT_N_DAYS = 21

# Test your data on the last N days
TEST_N_DAYS = 30

# TRAINING PARAMS
RANDOM_SEED = 36
TENSORBOARD_DIR = 'logs'
LEARNING_RATE = 1e-4
EPOCHS = 200 
MODEL_CHECKPOINT = 'best_rnn_model.hdf5'
TRAIN_LOSS = 'mae'
TRAIN_VAL_METRIC = 'val_loss'
VALIDATION_SPLIT = .2
BATCH_SIZE = 32

PREDICTION_FILE = 'time_series_prediction.tsv'
#####################################################################



### Model ###########################################################
REMOTE_FILE_PATH = '/home/ubuntu/time_series'
FTP_LOCATION = '13.250.31.141'
FTP_USER = 'ubuntu'
FTP_KEY_PATH = '/home/rohan/Desktop/d2i.pem'
#####################################################################
