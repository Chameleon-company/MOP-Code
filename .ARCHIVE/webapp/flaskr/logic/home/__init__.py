import pandas as pd
from sodapy import Socrata


"""
 Will search through all datasets available
 in Melbourne Open Data platform
 based on keyword matches in title, description...etc

 Returns a dataframe with the results
"""
def keyword_search(keywords=None):
    client = Socrata("data.melbourne.vic.gov.au",  # domain
                     'z99BiHe97JrarLpbpqRISffyr',  # app token
                     username="4l7fysec6unzmqbs1n9ulbdsz",  # api id
                     password="21kqwm0898v3yjkskdd2i940fc0g7quvr6rg80zalcyuhc1v4n")  # api secret
    rows = []
    for each_dataset in client.datasets():
        if each_dataset['resource']['name'].lower().find(keywords.lower()) != -1:
            rows.append([
                each_dataset['resource']['name'],
                each_dataset['resource']['id'],
                each_dataset['resource']['download_count'],
                each_dataset['permalink']
            ])

    df = pd.DataFrame(rows, columns=["Name", "Id", 'Downloads', 'Permalink'])

    return df
