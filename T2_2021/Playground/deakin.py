__verison__ = 1.0

import pandas as pd
from sodapy import Socrata


def summary_table():
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
    
        
        #print('download count',a['resource']['download_count'])
        _downloadCount.append(a['resource']['download_count'])
    
        _features.append(a['resource']['columns_name'])
        
    sum_df = pd.DataFrame({'Name':_name,
                    'id': id_,
                    'updatead_At':_updatedAt,
                    'createAt':_createAt,
                 #   'pageViews':_page_views,
               #     "attributes":_features,
                    "downloadCount":_downloadCount
                    })
    return sum_df, client

if __name__ == 'main':
    summary_table()


