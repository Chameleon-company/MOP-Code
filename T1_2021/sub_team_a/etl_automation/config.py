

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













### EXECUTION #######################################################

OUTPUT_DIR = 'op'
#####################################################################
