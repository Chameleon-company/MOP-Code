import os, sys
from datetime import datetime
import pandas as pd
from sodapy import Socrata

apptoken = "wMEkdLbVuXIpLiCFVic1PgiZ3"
domain = "data.melbourne.vic.gov.au"
client = Socrata(domain, apptoken, timeout=120)

# 029 "d6mv-s43h" pedestria counts/min in past hour (updated every 15min)
# 031 "h57g-5234" pedestrian count sensor locations

ds_c_id, ds_c_no = 'd6mv-s43h', "029"
ds_c_col = ['sensor_id', 'direction_1', 'direction_2', 'date_time']
folder = "/Users/albert.hon/playground/datasets"
ds_fname = os.path.join(folder, ds_c_no+"_"+ds_c_id+"__bufferred"+".csv")
ds_pq_fname = os.path.join(folder, ds_c_no+"_"+ds_c_id+"__bufferred"+".parquet")
log_fname = os.path.join(folder, ds_c_no+"_"+ds_c_id+"__log"+".txt")

def buffer_pedc():
    rc = client.get_all(ds_c_id)
    df_temp = pd.DataFrame.from_dict(rc)
    # write/append to csv
    if not os.path.isfile(ds_fname):
        df_temp[ds_c_col].to_csv(ds_fname, mode='w', header=True, index=False)  # first write
        sys.stdout = open(log_fname,'w') # write log
        print(f"{datetime.now()}, [pedestrian sensors counts] written to file")
    else:
        df_temp[ds_c_col].to_csv(ds_fname, mode='a', header=False, index=False)  # subsequent append to csv
        sys.stdout = open(log_fname,'a') # write log
        print(f"{datetime.now()}, [pedestrian sensors counts] appended to file")
    # trying parquet with gzip compression
    if not os.path.isfile(ds_pq_fname):
        df_temp[ds_c_col].to_parquet(ds_pq_fname, compression='gzip')  # first write pq
        sys.stdout = open(log_fname,'a')
        print(f"{datetime.now()}, [pedestrian sensors counts] written to parquet")
    else:  # this read and write parquet can snowball .. think of better way if data > 3 months
        df_prev_pq = pd.read_parquet(ds_pq_fname)
        df_new = pd.concat([df_prev_pq, df_temp[ds_c_col]], axis='index')
        df_new.to_parquet(ds_pq_fname, compression='gzip')  # subsequent update pq
        sys.stdout = open(log_fname,'a')
        print(f"{datetime.now()}, [pedestrian sensors counts] updated to parquet")

if __name__ == '__main__':
    buffer_pedc()

# this buffering is set to run once an hour
# and I think there may be duplicates
# after buffering for some time --> check this
# if this needs to be solved, can either 
# [1] have a regular script to read csv, remove duplicates, write to csv ?
# [2] have a regular script to read csv, remove duplicates, and separate into equal size time interval files