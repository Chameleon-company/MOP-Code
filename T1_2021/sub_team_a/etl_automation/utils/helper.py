import requests
import os

def download_zip(url, save_path, chunk_size=128):
	headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
		}
	r = requests.get(url, stream=True, headers=headers)
	with open(save_path, 'wb') as fd:
		for chunk in r.iter_content(chunk_size=chunk_size):
			fd.write(chunk)




def create_dir(dir_path):
	if not os.path.exists(dir_path):
		os.mkdir(dir_path)