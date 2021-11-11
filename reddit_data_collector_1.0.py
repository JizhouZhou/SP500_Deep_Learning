import requests
import pandas as pd
from datetime import datetime

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

reddit_credential = pd.read_csv('reddit_credential.csv', header = None)[0]

auth = requests.auth.HTTPBasicAuth(reddit_credential[0], reddit_credential[1])

data = {
        'grant_type': 'password',
        'username': reddit_credential[2],
        'password': reddit_credential[3]
        }

headers = {'User-Agent': 'Zhou'}

# send our request for an OAuth token
res = requests.post('https://www.reddit.com/api/v1/access_token',
                    auth=auth, data=data, headers=headers)

# convert response to JSON and pull access_token value
TOKEN = res.json()['access_token']

# add authorization to our headers dictionary
headers = {**headers, **{'Authorization': f"bearer {TOKEN}"}}

# while the token is valid (~2 hours) we just add headers=headers to our requests
requests.get('https://oauth.reddit.com/api/v1/me', headers=headers)

# we use this function to convert responses to dataframes
def df_from_response(res):
    # initialize temp dataframe for batch of data in response
    df = pd.DataFrame()

    # loop through each post pulled from res and append to df
    for post in res.json()['data']['children']:
        df = df.append({
            'subreddit': post['data']['subreddit'],
            'title': post['data']['title'],
            'selftext': post['data']['selftext'],
            'upvote_ratio': post['data']['upvote_ratio'],
            'ups': post['data']['ups'],
            'downs': post['data']['downs'],
            'score': post['data']['score'],
            'link_flair_css_class': post['data']['link_flair_css_class'],
            'created_utc': datetime.fromtimestamp(post['data']['created_utc']).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'id': post['data']['id'],
            'kind': post['kind']
        }, ignore_index=True)

    return df

data = pd.DataFrame()
params = {'limit': 100,
          'start_date': datetime(2010, 5, 6),
          'end_date': datetime(2010,5,9)}

# loop through 10 times (returning 1K posts)
for i in range(1):
    # make request
    res = requests.get("https://oauth.reddit.com/r/finance",
                       headers=headers,
                       params=params)

    # get dataframe from response
    new_df = df_from_response(res)
    # take the final row (oldest entry)
    row = new_df.iloc[len(new_df)-1]
    # create fullname
    fullname = row['kind'] + '_' + row['id']
    # add/update fullname in params
    params['after'] = fullname
    
    # append new_df to data
    data = data.append(new_df, ignore_index=True)