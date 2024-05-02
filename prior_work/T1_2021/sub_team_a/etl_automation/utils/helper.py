import requests
import os
import zipfile
import glob
from shutil import move, rmtree
import pandas as pd


COLUMNS_UPDATE_DICT = {
						'year':'Year','month':'Month','mdate':'Day'
					}


def download_file(url, save_path, chunk_size=128):
	"""
	Download file from param
	"""
	headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36',
		}
	r = requests.get(url, stream=True, headers=headers)
	with open(save_path, 'wb') as fd:
		for chunk in r.iter_content(chunk_size=chunk_size):
			fd.write(chunk)


def create_dir(dir_path):
	"""
	Create dir if it doesn't exists
	"""
	if not os.path.exists(dir_path):
		os.makedirs(dir_path)


def extract_zip_dir(zip_dir_path, unzip_dir):
	"""
	Unzip a zip file
	"""
	with zipfile.ZipFile(zip_dir_path, 'r') as zip_ref:
		zip_ref.extractall(unzip_dir)


def get_file_list_from_dir(dir_path, ext):
	"""
	Returns list of files of the required extension from a directory
	"""
	return [os.path.join(dir_path, filename) for filename in os.listdir(dir_path) if filename.endswith(ext)]


def move_file(src_path, dest_path):
	move(src_path, dest_path)


def remove_dir(dir_path):
	rmtree(dir_path)


def read_csv_file(file_path, delim=','):
	return pd.read_csv(file_path, delim)


def get_datetime_column(df, col):
	df[col] = pd.to_datetime(df[col])
	return df


def convert_to_datetime(df, feat_cols, date_col):

	df.rename(
				columns=COLUMNS_UPDATE_DICT, 
				inplace=True
			)

	df[date_col] = df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-' + df['Day'].astype(str) 
	df = get_datetime_column(df,date_col)
	required_feat_cols = list(set(feat_cols).intersection(set(df.columns.tolist())))
	df = df[[date_col,required_feat_cols[0]]]
	return df


def merge_files(df1, df2, merge_type, left_on, right_on):
	return pd.merge(df1, df2, how=merge_type, left_on=left_on, right_on=right_on)


def get_dataframe_from_json(data):
	return pd.DataFrame.from_records(data)