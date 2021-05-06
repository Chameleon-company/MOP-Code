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
	'rainfall': -1490874013,
	'max_temp' : -1490871123,
	'min_temp' : -1490871319,
	'solar_exposure' : -1490889015

}

# BOM DOWNLOAD URL PATTERN
BOM_DOWNLOAD_BASE_URL = 'http://www.bom.gov.au/jsp/ncc/cdio/weatherData/av?p_display_type=dailyZippedDataFile&p_stn_num={}&p_c={}&p_nccObsCode={}&p_startYear='

EXTRACTION_DIR = 'extraction'
#####################################################################



### Tranform ########################################################

TRANSFORMATION_DIR = 'transform'

FEAT_COL_NAME = {
	'rainfall': 'Rainfall amount (millimetres)',
	'max_temp' : 'Maximum temperature (Degree C)',
	'min_temp' : 'Minimum temperature (Degree C)',
	'solar_exposure' : 'Daily global solar exposure (MJ/m*m)'
}

DATE_COLUMN = 'date'

MERGED_FEATURES_FILE = 'features_merged.csv'

# USE BOM DATA AFTER THIS DATE
BOM_CUTOFF_DATE = '2015-01-01'



HOLIDAY_FILE_PATH = 'user_input/holidays.tsv'
HOLIDAY_COLUMN_NAME = 'Public_Holiday'


COVID_RESTRICTION_FILE_PATH = 'user_input/restriction.tsv'
COVID_RESTRICTION_COLUMN_NAME = 'Covid Restrictions'
#####################################################################




