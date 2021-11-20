import os, sys
from datetime import datetime
import pandas as pd
from sodapy import Socrata

apptoken = None
domain = "data.melbourne.vic.gov.au"
client = Socrata(domain, apptoken, timeout=120)

# 029 "d6mv-s43h" pedestria counts/min in past hour (updated every 15min)
# 031 "h57g-5234" pedestrian count sensor locations

ds_l_id, ds_l_no = 'h57g-5234', "031"
folder = "/Users/albert.hon/playground/datasets"
ds_fname = os.path.join(folder, ds_l_no+"_"+ds_l_id+"__list"+".csv")
log_fname = os.path.join(folder, ds_l_no+"_"+ds_l_id+"__log"+".txt")

def buffer_pedl():
    rl = client.get_all(ds_l_id)
    df_temp = pd.DataFrame.from_dict(rl)
    if not os.path.isfile(ds_fname):
        df_temp.to_csv(ds_fname, mode='w', header=True, index=False)  # first write
        sys.stdout = open(log_fname,'w') # write log
        print(f"{datetime.now()}, [pedestrian sensors locations list] written to file")
    else:
        df_temp_prev = pd.read_csv(ds_fname, dtype=str)
        df_temp_new = pd.concat(
            [df_temp_prev, df_temp], axis='index').drop_duplicates(subset=['sensor_id','installation_date'])
        df_temp_new.to_csv(ds_fname, mode='w', header=True, index=False)
        sys.stdout = open(log_fname,'a') # write log
        print(f"{datetime.now()}, [pedestrian sensors locations list] updated to file")

if __name__ == '__main__':
    buffer_pedl()