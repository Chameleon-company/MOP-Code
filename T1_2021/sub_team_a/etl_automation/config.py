
# Olympia PARK
STATION_CODE = '086338'

FEAT_NCCOBS_CODES = {
	'rainfall': 136,
	'temperature' : 122,
	'solar_exposure' : 193

}

FEAT_PC_CODES = {
	'rainfall': -1490874013,
	'temperature' : -1490871123,
	'solar_exposure' : -1490889015

}



# BOM DOWNLOAD URL PATTERN
BOM_DOWNLOAD_BASE_URL = 'http://www.bom.gov.au/jsp/ncc/cdio/weatherData/av?p_display_type=dailyZippedDataFile&p_stn_num={}&p_c={}&p_nccObsCode={}&p_startYear='
