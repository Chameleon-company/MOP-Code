import deakin 
import sys 
import os 
import pandas as pd 
import argparse
import requests

parser = argparse.ArgumentParser(description='''Play with datasets from https://data.melbourne.vic.gov.au/''')
parser.add_argument('-filter',type=int, help='0: inspect popular datasets, 1:search dataset, 2: download type_I, 3: download type_II')
parser.add_argument('-n', type=str, help='''filter=0, no input;
                                            filter=1, typing keyword;
                                            filter=2, typing id;
                                            filter=3, typing multiple dataset names.
                                            ''')
parser.add_argument('-limit',type=int, help='''filter=0, #_of datasets;
                                                filter= 2 or 3, #of data size''')

args = parser.parse_args()

#download_type_I: downloading all datasets based on input
table, client= deakin.top_table() #init object 


if args.filter == 0:
    table, client= deakin.top_table(args.limit)
    print(table)
    print('\n\n********Note********')
    print('https://data.melbourne.vic.gov.au/d/$id')
    print('https://data.melbourne.vic.gov.au/api/catalog/v1?ids=$id')
    print('''We encourage users to check dataset according to the link of each data.
There are many detailed descriptions on the official website of city of melbourne.''')


if args.filter == 1:
    data_table = deakin.data_inspect(keywords=args.n,client = client)
    print(data_table)
    print('\n\n********Note********')
    print('https://data.melbourne.vic.gov.au/d/$id')
    print('https://data.melbourne.vic.gov.au/api/catalog/v1?ids=$id')
    print('''We encourage users to check dataset according to the link of each data.
There are many detailed descriptions on the official website of city of melbourne.''')


if args.filter == 2:
    pref_path = os.getcwd()+'/Datasets'
    if args.n[0] == '_':
        args.n = args.n[1:]

    if args.limit is not None:
        try:
            result  =client.get(args.n,limit=args.limit)
        except requests.exceptions.HTTPError as err:
            print('\n********************plz give correct ID :',err)
            sys.exit(0)
    else:
        try:    
            result  =client.get(args.n,limit=5000)
        except requests.exceptions.HTTPError as err:
            print('\n********************plz give correct ID :',err)
            sys.exit(0)

    name = client.get_metadata(args.n)['name']+'.csv'
    results_df = pd.DataFrame.from_records(result)
    results_df = results_df.iloc[:,:-1]
    try:
        os.makedirs(pref_path)
    except OSError:
        pass
    else:
        print ("Successfully created the directory %s" % path)

    results_df.to_csv(os.path.join(pref_path,name),index=False)
    print("Successfully save in directory %s" %os.path.join(pref_path,name))



    




