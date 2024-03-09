import os
import time
from datetime import datetime
import pandas as pd
from sodapy import Socrata

apptoken = os.environ.get("wMEkdLbVuXIpLiCFVic1PgiZ3")
domain = "data.melbourne.vic.gov.au"
client = Socrata(domain, apptoken)

ds_id, ds_no = 'vh2v-4nfs', "003"
ds_fname = os.path.join('datasets', ds_no+"_"+ds_id+"__bufferred"+".csv")
uq_fname = os.path.join('datasets', ds_no+"_"+ds_id+"__uniqueBays"+".csv")
ds_col1 = ['bay_id', 'st_marker_id', 'status', 'lat', 'lon']
ds_col2 = ['bay_id', 'st_marker_id', 'lat', 'lon']
log_fname = os.path.join('datasets', ds_no+"_"+ds_id+"__log"+".txt")

r1 = client.get_all(ds_id)  # get snapshot of dataset using sodapy api
df_temp = pd.DataFrame.from_dict(r1)  # read dict to dataframe
df_temp1 = df_temp[ds_col1]
df_temp2 = df_temp[ds_col2]
df_temp1['db_read_time'] = datetime.now()  # add timestampe column

df_temp1.to_csv(ds_fname, mode='a', header=False, index=False)  # subsequent append to csv
df_temp2prev = pd.read_csv(uq_fname)
df_temp2new = pd.concat([df_temp2prev, df_temp2], axis='index').drop_duplicates()
df_temp2new.to_csv(uq_fname, mode='w', header=True, index=False)
o = open(log_fname,'a')
print(f"{datetime.now()}, dataset retrieved and appended to csv", file=o)
print(f"{datetime.now()}, unique parking bays csv updated", file=o)
o.close()
