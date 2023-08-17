import json
import xml.etree.ElementTree as ET
import math
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

def dataset_analysis(dataset_name):
    q_list = []
    q_click = []
    q_dwell = []
    q_move = []
    q_user = []
    
    path = f'{dataset_name}'
    
    if dataset_name == 'TianGong':
        with open(f'{path}/sessions_v5.json', encoding='UTF-8') as f:
            data = json.load(f)
        
        for i in range(len(data)):
            queries = data[i]["queries"]
            query = []
            clicked = []
            q_dw = []
            q_dist = []
            user = []
            
            for j in range(len(queries)):
                query.append(queries[j]["query_string"])
                SERPs = queries[j]["SERPs"]
                results = SERPs[0]["results"]
                move = SERPs[0]["mouse_moves"]
                user.append(data[i]['user_id'])
                c = sum(1 for result in results if result["clicked"] == 1)
                clicked.append(c)
                dw = dist = 0
                if move:
                    dw = move[-1]['Et'] - move[0]['St']
                    dist = sum(math.sqrt((m['Sx'] - m['Ex'])**2 + (m['Sy'] - m['Ey'])**2) for m in move if 'Ex' in m)
                q_dw.append(dw)
                q_dist.append(dist)

            q_list.append(query)
            q_click.append(clicked)
            q_dwell.append(q_dw)
            q_move.append(q_dist)
            q_user.append(user)
    
    elif dataset_name == 'KDD19':
        df = pd.read_csv(f'{path}/anno_log.csv')
        df['session'] = df['studentID'].astype('string') + '-' + df['task_id'].astype('string')
        s_list = list(set(df['session']))
        
        for s in s_list:
            s_df = df[df['session'] == s]
            q = list(set(s_df['query']))
            q = [i for i in q if i != ' ']
            clk = []
            q_dw = []
            q_dist = []
            user = []
            
            for i in q:
                q_df = s_df[s_df['query'] == i]
                clk.append(len(q_df[q_df['action'] == 'CLICK']))
                q_time = q_df['content'].str.split('\t').str[0].str.replace('TIME=', '')
                dw = int(q_time.iloc[-1]) - int(q_time.iloc[0])
                q_dw.append(dw)
                q_xy = q_df['content'].str.split('INFO').str[1].str.split('\t')
                q_xy = q_xy[q_xy.str[6].notna()]
                dist = np.sqrt(
                    (q_xy.str[2].str.replace('x=', '').astype('float') - q_xy.str[5].str.replace('x=', '').astype('float'))**2 +
                    (q_xy.str[3].str.replace('y=', '').astype('float') - q_xy.str[6].str.replace('y=', '').astype('float'))**2
                ).sum()
                q_dist.append(dist)
                user.append(q_df['studentID'].iloc[0])

            q_list.append(q)
            q_click.append(clk)
            q_dwell.append(q_dw)
            q_move.append(q_dist)
            q_user.append(user)
    
    elif dataset_name == 'track2014':
        tree = ET.parse(f'{path}/sessiontrack2014.xml')
        root = tree.getroot()
        
        for session in root.iter('session'):
            st = []
            q = []
            clk = []
            user = []
            
            for interaction in session.iter('interaction'):
                st.append(interaction.attrib['starttime'])
                q.append(interaction[0].text)
                user.append(session.attrib['userid'])
                c = sum(1 for _ in interaction.iter('click'))
                clk.append(c)
            
            q_list.append(q)
            q_click.append(clk)
            q_user.append(user)
            
            cur_time = '0'
            if session[-1].tag == 'currentquery':
                cur_time = session[-1].attrib['starttime']
            q_dwell.append([float(cur_time) - float(st[i]) for i in range(len(st) - 1)])

    return q_list, q_click, q_dwell, q_move, q_user

# Test for one dataset to see if the function works
q_list_tiangong, q_click_tiangong, q_dwell_tiangong, q_move_tiangong, q_user_tiangong = dataset_analysis('TianGong')
len(q_list_tiangong), len(q_click_tiangong), len(q_dwell_tiangong), len(q_move_tiangong), len(q_user_tiangong)


def pre_analysis(dataset_name):
    q_list, q_click, q_dwell, q_move, q_user = dataset_analysis(dataset_name)
    
    # Extracting query lengths
    def queryLength(query_list):
        all_len = []
        for session in query_list:
            query_len = []
            for query in session:
                en_num_count = other_count = 0
                count = 0
                for word in query:
                    if 'a' <= word <= 'z' or 'A' <= word <= 'Z' or word.isdigit():
                        count = 1
                    else:
                        other_count += 1
                    en_num_count += count
                    count = 0
                word_count = en_num_count + other_count
                query_len.append(word_count)
            all_len.append(query_len)
        return all_len
    
    q_len = queryLength(q_list)
    
    sess_max = max(len(q) for q in q_len)
    fea_df = (pd.DataFrame(q_len, columns=range(1,sess_max+1)).astype('string') +
              ',' + pd.DataFrame(q_click, columns=range(1,sess_max+1)).astype('string') +
              ',' + pd.DataFrame(q_dwell, columns=range(1,sess_max+1)).astype('string') +
              ',' + pd.DataFrame(q_move, columns=range(1,sess_max+1)).astype('string') +
              ',' + pd.DataFrame(q_user, columns=range(1,sess_max+1)).astype('string'))
    fea_session = pd.DataFrame(pd.concat([fea_df.index.to_series()]*sess_max, axis=1), columns=range(1,sess_max+1))
    fea_query = pd.concat([pd.DataFrame([fea_df.columns], columns=range(1,sess_max+1))]*len(fea_df), axis=0, ignore_index=True)
    total_list = (fea_df.astype('string') + ',' + 
                fea_session.astype('string') + ',' + 
                fea_query.astype('string'))
    total_list = [[y for y in x if pd.notna(y)] for x in total_list.values.tolist()]
    total_list = [item for sublist in total_list for item in sublist]
    total_df = pd.DataFrame([sub.split(",") for sub in total_list], 
                            columns=['query_length','query_click','query_dwell','query_move','userID','sessionID','queryID'])
    total_df = total_df.astype('float')
    total_df[['query_length','query_click','userID','sessionID','queryID']] = total_df[
        ['query_length','query_click','userID','sessionID','queryID']].astype('int')
    total_df['query_step'] = 0
    for session in set(total_df['sessionID']):
        total_df.loc[total_df.sessionID==session,'query_step'] = (total_df.loc[total_df['sessionID']==session,'queryID']/
                              len(total_df.loc[total_df['sessionID']==session]))
    #%%
    feas = ['query_length','query_click','query_dwell','query_move']
    nor_feas = ['nor_length','nor_click','nor_dwell','nor_move']
    total_df[nor_feas] = np.log2(total_df[feas]+1)
    for user in set(total_df['userID']):
        user_df = total_df.loc[total_df['userID']==user].copy()
        nor_user = stats.zscore(user_df[nor_feas], nan_policy='omit')
        nor_user = np.nan_to_num(nor_user)
        total_df.loc[total_df.userID==user, nor_feas] = nor_user
    nor_fea = total_df[nor_feas]
    kmeans = KMeans(n_clusters=4, random_state=0).fit(nor_fea)
    c_labels = kmeans.labels_
    total_df['cluster_label'] = c_labels
    #%%
    total_df['cluster_label']=total_df['cluster_label'].replace(4,0)
    sns.pairplot(total_df[nor_feas+['cluster_label']],
                 hue='cluster_label',palette='tab10')
    plt.show()
    
# Testing pre_analysis for "TianGong" (this will throw a FileNotFoundError but should demonstrate the structure)
try:
    pre_analysis("TianGong")
except FileNotFoundError:
    pass