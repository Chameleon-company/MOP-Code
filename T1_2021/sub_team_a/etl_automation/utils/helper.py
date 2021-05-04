import requests
import os
import zipfile
import glob
from shutil import move, rmtree

def download_file(url, save_path, chunk_size=128):
	"""
	Download file from param
	"""
	headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
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