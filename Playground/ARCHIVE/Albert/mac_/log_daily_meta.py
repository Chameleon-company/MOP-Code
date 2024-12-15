import os
import sys
from datetime import datetime
import pandas as pd
from d2i_tools2 import getMeta

folder = "/Users/albert.hon/playground/datasets"
meta_fname, track_fname = "meta_log.csv", "meta_log_track.txt"
meta_log = os.path.join(folder, meta_fname)
track_log = os.path.join(folder, track_fname)

# function to call from crontab to read CoM dataset meta data to create a 'time history' of
# meta data like page downloads / page views that would otherwise not be available


def log_meta(apptoken=None):
    metadf = getMeta(apptoken)
    metadf['log_time'] = datetime.now()
    log_cols = ['id', 'data_upd_at', 'pv_last_wk',
                'pv_last_mth', 'pv_total', 'download_count', 'log_time']
    if not os.path.isfile(meta_log):
        metadf[log_cols].to_csv(
            meta_log, mode='w', header=True, index=False)  # first write
        sys.stdout = open(track_log, 'w')  # write log
        print(f"{datetime.now()}, [meta log] written to file")
    else:
        metadf[log_cols].to_csv(
            meta_log, mode='a', header=False, index=False)  # subsequent writes
        sys.stdout = open(track_log, 'a')  # write log
        print(f"{datetime.now()}, [meta log] appended to file")


if __name__ == '__main__':
    log_meta()
