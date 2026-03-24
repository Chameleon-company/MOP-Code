import paramiko
import os

def copy_ftp_file_over_sftp_using_key(cfg):
	"""
	ftp file copy over sftp using key based authentication
	"""
	prev_stage_output = os.path.join(cfg.OUTPUT_DIR, cfg.TRANSFORMATION_DIR, cfg.PREDICTION_FILE)
	local_src_file_loc = prev_stage_output 
	remote_dest_file_loc = os.path.join(cfg.REMOTE_FILE_PATH, cfg.PREDICTION_FILE)
	ssh_client=paramiko.SSHClient()
	ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())			# enable the host machine to trust remote machine
	ssh_client.connect(hostname=cfg.FTP_LOCATION,username=cfg.FTP_USER,key_filename=cfg.FTP_KEY_PATH)
	# file copy
	ftp_client=ssh_client.open_sftp()

	ftp_client.put(local_src_file_loc, remote_dest_file_loc)
	ftp_client.close()

