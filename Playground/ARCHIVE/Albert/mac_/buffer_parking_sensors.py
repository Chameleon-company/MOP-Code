import os, sys
from datetime import datetime
import pandas as pd
from sodapy import Socrata

apptoken = "wMEkdLbVuXIpLiCFVic1PgiZ3"
domain = "data.melbourne.vic.gov.au"
client = Socrata(domain, apptoken, timeout=120)

ds_id, ds_no = 'vh2v-4nfs', "003"

folder = "/Users/albert.hon/playground/datasets"
ds_fname = os.path.join(folder, ds_no+"_"+ds_id+"__bufferred"+".csv")
uq_fname = os.path.join(folder, ds_no+"_"+ds_id+"__uniqueBays"+".csv")
ds_pq_fname = os.path.join(folder, ds_no+"_"+ds_id+"__bufferred"+".parquet")
ds_col1 = ['bay_id', 'st_marker_id', 'status', 'lat', 'lon']
ds_col2 = ['bay_id', 'st_marker_id', 'lat', 'lon']
log_fname = os.path.join(folder, ds_no+"_"+ds_id+"__log"+".txt")

def buffer_ps():
    r1 = client.get_all(ds_id)  # get snapshot of dataset using sodapy api
    df_temp = pd.DataFrame.from_dict(r1)  # read dict to dataframe
    df_temp1 = df_temp[ds_col1].copy()
    df_temp2 = df_temp[ds_col2].copy()
    df_temp1['db_read_time'] = datetime.now()  # add timestampe column
    # write/append to csv
    if not os.path.isfile(ds_fname):
        df_temp1.to_csv(ds_fname, mode='w', header=True,
                        index=False)  # first write
        df_temp2.to_csv(uq_fname, mode='w', header=True, index=False)
        # write log
        sys.stdout = open(log_fname,'w')
        print(f"{datetime.now()}, [parking sensors status] and [unique bays] written to files")
    else:
        df_temp1.to_csv(ds_fname, mode='a', header=False,
                        index=False)  # subsequent append to csv
        df_temp2prev = pd.read_csv(uq_fname)
        df_temp2new = pd.concat(
            [df_temp2prev, df_temp2], axis='index').drop_duplicates(subset=['st_marker_id'])
        df_temp2new.to_csv(uq_fname, mode='w', header=True, index=False)
        # write log
        sys.stdout = open(log_fname,'a')
        print(f"{datetime.now()}, [parking sensors status] appended and [unique bays] updated to files")
    # trying parquet with gzip compression
    if not os.path.isfile(ds_pq_fname):
        df_temp1.to_parquet(ds_pq_fname, compression='gzip')  # first write pq
        sys.stdout = open(log_fname,'a')
        print(f"{datetime.now()}, [parking sensors status] written to parquet")
    else:  # this read and write parquet can snowball .. think of better way if data > 3 months
        df_prev_pq = pd.read_parquet(ds_pq_fname)
        df_new = pd.concat([df_prev_pq, df_temp1], axis='index')
        df_new.to_parquet(ds_pq_fname, compression='gzip')  # subsequent update pq
        sys.stdout = open(log_fname,'a')
        print(f"{datetime.now()}, [parking sensors status] updated to parquet")


if __name__ == '__main__':
    buffer_ps() 