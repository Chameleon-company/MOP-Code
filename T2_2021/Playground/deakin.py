__verison__ = 1.0

import pandas as pd
from sodapy import Socrata


def top_table(top = 10):

    ''' return top # downloads of datasets  '''
    client = Socrata("data.melbourne.vic.gov.au", #domain
                'z99BiHe97JrarLpbpqRISffyr',  #app token
                username="4l7fysec6unzmqbs1n9ulbdsz", #api id
                password="21kqwm0898v3yjkskdd2i940fc0g7quvr6rg80zalcyuhc1v4n") #api secrete

    _name = []
    id_ = []
    _updatedAt = []
    _createAt = []
    _page_views = []
    _downloadCount = []
    _features = []
    _link = []
    _view = []
    
    
    for a in client.datasets(): #222
        
        #print('dataset_name',a['resource']['name'])
        _name.append(a['resource']['name'])
        #print('unique identifier',a['resource']['id'])
        id_.append(a['resource']['id'])
    
        #print('updatedAt',a['resource']['updatedAt'])
        _updatedAt.append(a['resource']['updatedAt'])
    
        #print('createAt', a['resource']['createdAt'])
        _createAt.append(a['resource']['createdAt'])
        
        #print('page_views', a['resource']['page_views'])
        _page_views.append(a['resource']['page_views'])
        
        #print('features name', a['resource']['columns_name'])
    
        _link.append(a['permalink'])
        #print('download count',a['resource']['download_count'])
        _downloadCount.append(a['resource']['download_count'])
    
        _features.append(a['resource']['columns_name'])
        
        _view.append(a['resource']['page_views']['page_views_last_month'])
    sum_df = pd.DataFrame({'Name':_name,
                    'id': id_,
                    #'updatead_At':_updatedAt,
                    #'createAt':_createAt,
                    "link": _link,
                 #   'pageViews':_page_views,
               #     "attributes":_features,
                    "downloadCount":_downloadCount,
                    'page_views_last_month':_view
                    })

    sum_df = sum_df.sort_values(['downloadCount'], ascending=False).head(top)
    sum_df = sum_df.reset_index().drop('index', axis=1)
    return sum_df, client

def keyword_search(keywords= None):
    if keywords is None:
        print('Please typing valid dataset name')
    else:
        sum_table, client= top_table()
        rows = []
        for each_dataset in client.datasets():
        #for key_name in each_dataset['resource']:
            #    print(key_name,":" ,each_dataset['resource'][str(key_name)])    
            if each_dataset['resource']['name'].lower().find("street".lower()) != -1:
            # print('name: ' , each_dataset['resource']['name'],', id:' , each_dataset['resource']['id'])
                rows.append([each_dataset['resource']['name'], 
                each_dataset['resource']['id'],
                each_dataset['resource']['download_count'],
                each_dataset['permalink']
                ]
                )
        df = pd.DataFrame(rows, columns=["Name", "Id", 'Downloads','link'])

        return df



if __name__ == 'main':

    keyword_search('parking')
    print(df)

