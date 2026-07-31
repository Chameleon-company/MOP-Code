import pandas as pd
from sodapy import Socrata
import seaborn as sns 
from matplotlib import pyplot as plt 

def top_table(top = 20):

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
    _categorial = []
    _tag = []
    
    
    for a in client.datasets(): #222
        
        #print('dataset_name',a['resource']['name'])
        _name.append(a['resource']['name'])
        #print('unique identifier',a['resource']['id'])
        id_.append(a['resource']['id'])
    
        #print('updatedAt',a['resource']['updatedAt'])
        _updatedAt.append(a['resource']['updatedAt'])
    
        #print('createAt', a['resource']['createdAt'])
        _createAt.append(a['resource']['createdAt'])

        _page_views.append(a['resource']['page_views'])
        
        #print('features name', a['resource']['columns_name'])
        _link.append(a['permalink'])
        #print('download count',a['resource']['download_count'])
        _downloadCount.append(a['resource']['download_count'])
    
        _features.append(a['resource']['columns_name'])
        
        _view.append(a['resource']['page_views']['page_views_last_month'])
        _categorial.append(a['classification']['domain_category'])
        _tag.append(a['classification']['domain_tags'])

    sum_df = pd.DataFrame({'Name':_name,
                    'id': id_,
                    #'updatead_At':_updatedAt,
                    #'createAt':_createAt,
            #        "link": _link,
            #       'pageViews':_page_views,
            #     "attributes":_features,
                    "downloadCount":_downloadCount,
                    'page_views_last_month':_view,
                    'categorical':_categorial
                    })

    sum_df = sum_df.sort_values(['downloadCount'], ascending=False).head(top)
    sum_df = sum_df.reset_index().drop('index', axis=1)
    return sum_df, client

# for web.app use
def keyword_search(keywords= None):
    if keywords is None:
        print('Please typing valid dataset name')
    else:
        sum_table, client= top_table()
        rows = []
        for each_dataset in client.datasets():
            check_list = []
            for i in keywords.lower().split():
                if each_dataset['resource']['name'].lower().find(i) != -1:
                    check_list.append(1)
                else:
                    check_list.append(0)
            if any(check_list):
                rows.append([each_dataset['resource']['name'], 
                each_dataset['resource']['id'],
                each_dataset['resource']['download_count'],
                each_dataset['classification']['domain_category'],
                each_dataset['classification']['domain_tags'],
                each_dataset['resource']['page_views']['page_views_last_month']
            ])
        df = pd.DataFrame(rows, columns=["Name", "Id", 'Downloads','Categorical','Tags','page_views_last_month'])
        df = df.sort_values(['page_views_last_month'], ascending=False)
        df = df.reset_index().drop('index', axis=1)
        
        fig, ax = plt.subplots(dpi=60, figsize=(9,11))
        
        ax = sns.barplot(y = df['Name'], x = df['page_views_last_month'], palette=sns.color_palette("vlag"))
        ax.set_xlabel('page_views_last_month', fontsize = 18)
        ax.set_ylabel('Name', fontsize = 18)
        ax.set(xlim=(0,df['page_views_last_month'].max()))
        plt.close()
        
        return fig, df

def data_inspect(keywords= None, client=None):
    if keywords is None:
        print('Please typing valid dataset name')
    else:
        rows = []
        for each_dataset in client.datasets():
            check_list = []
            for i in keywords.lower().split():
                if each_dataset['resource']['name'].lower().find(i) != -1:
                    check_list.append(1)
                else:
                    check_list.append(0)
            if any(check_list):
                rows.append([each_dataset['resource']['name'], 
                each_dataset['resource']['id'],
                each_dataset['resource']['download_count'],
                each_dataset['classification']['domain_category'],
                each_dataset['classification']['domain_tags']
            ])
        df = pd.DataFrame(rows, columns=["Name", "Id", 'Downloads','Categorical','Tags'])
        df = df.sort_values(['Downloads'], ascending=False)
        df = df.reset_index().drop('index', axis=1)
        return df



#if __name__ == 'main':


