import csv
import json
import click
import os
import logging
import random
import pandas as pd
import numpy as np
import clustering as cl
import scipy.spatial.distance as  ssd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from twarc import ensure_flattened
from io import TextIOWrapper

logging.getLogger().setLevel(logging.INFO)
TEMPORAL_CSV_PATH = 'output.csv'


def generate_random_file_name():
    return '.' + str(random.randint(0, 20000)) + '_' + TEMPORAL_CSV_PATH


def set_score_value(username, score, dictionary):
    dictionary[username] = score


def get_score_value(username, dictionary):
    return dictionary[username]


def compute_score(username, count, alpha, dictionary):
    score = count + alpha * get_score_value(username, dictionary) if alpha > 0 else count
    set_score_value(username, score, dictionary)
    return score


@click.command()
@click.option('-a', '--alpha', type=click.FLOAT, required=False, default='0.005')
@click.option('-g', '--granularity', required=False, default='M', type=click.STRING)
@click.option('-t', '--threshold', required=False, default='2.0', type=click.FLOAT)
@click.option('-i', '--interval', required=False, type=click.STRING)
@click.option('-m', '--method', type=click.STRING, default = '-')
@click.option('-l', '--algorithm', type=click.STRING, default = '-')
@click.argument('infile', type=click.File('r'), default='-')
@click.argument('outfile', type=click.STRING, default='-')

def main(infile: TextIOWrapper,
         outfile: str,
         alpha: float,
         threshold: float,
         granularity: str,
         interval: str,
         method: str,
         algorithm: str):

    first_temporal_csv = generate_random_file_name()

    f_temp_output = open(first_temporal_csv, 'w', encoding="utf-8")
    f_temp_output.write("created_at,author_name,author_profile\n")

    profile_image_dictionary = dict()
    logging.info('Generating temporal output file: ' + first_temporal_csv)
    is_interval = False
    start_time = None
    end_time = None

    if interval is not None:
        is_interval = True
        splitted_time = interval.split(',')
        start_time = pd.to_datetime(splitted_time[0], utc=True)
        end_time = pd.to_datetime(splitted_time[1], utc=True)

    for line in infile:
        for tweet in ensure_flattened(json.loads(line)):
            if 'referenced_tweets' in tweet:
                for x in tweet['referenced_tweets']:
                    if 'retweeted' in x['type']:
                        author_name = x['author']['username']
                        author_profile = x['author']['profile_image_url']
                        created_at = tweet['created_at']
                        profile_image_dictionary[author_name] = author_profile
                        is_allowed = True
                        if is_interval:
                            created_at_time = pd.to_datetime(created_at, utc=True)
                            if not start_time <= created_at_time <= end_time:
                                is_allowed = False
                        if is_allowed:
                            f_temp_output.write("{},{},{}\n".format(created_at, author_name, author_profile))
    f_temp_output.close()

    logging.info('Temporal file generated.')
    df = pd.read_csv(first_temporal_csv)
    os.remove(first_temporal_csv)
    if len(df.index) == 0:
        logging.info("No users to process")
        return
    df = df.dropna()
    df['created_at'] = pd.to_datetime(df['created_at']).dt.to_period(granularity)
    unique_dates = list(df.created_at.unique())
    unique_dates.sort()
    unique_usernames = set(df.author_name.unique())

    dictionary_periods = dict()
    for username in unique_usernames:
        dictionary_periods[username] = 0


    second_temporal_csv = generate_random_file_name()
    f_temp_output = open(second_temporal_csv, 'w', encoding="utf-8")
    f_temp_output.write("profile_image_url,author_name")
    for unique_date in unique_dates:
        unique_date_str: str = str(unique_date).split('/')[0]
        f_temp_output.write(',' + str(unique_date_str).replace(' ', '_'))
    f_temp_output.write("\n")

    logging.info('Computing user scores')
    user_count: int = 1
    total_users: int = len(unique_usernames)
    for user in unique_usernames:
        f_temp_output.write(profile_image_dictionary[user])
        f_temp_output.write(","+user)
        df_filtered_user = df[df['author_name'] == user]
        for date_period in unique_dates:
            df_filtered = df_filtered_user[df_filtered_user['created_at'] == date_period]
            number_of_rts = len(df_filtered.index)
            score = compute_score(user, number_of_rts, alpha, dictionary_periods)
            f_temp_output.write("," + str(score))
        f_temp_output.write("\n")
        logging.info('{}/{}'.format(user_count, total_users))
        user_count = user_count + 1
    f_temp_output.close()

    logging.info('User scores computed. Matrix file generated:' + outfile)
    logging.info('Filtering users...')

    connectivity ={}

    f_temp_input = open(second_temporal_csv, 'r', encoding="utf-8")
    f_output = open(outfile, 'w', encoding='utf-8')
    csv_file = csv.reader(f_temp_input)
    number_of_line = 0
    for line in csv_file:
        if number_of_line > 0:
            sum_score = sum([float(x) for x in line[2:]])
            if sum_score >= threshold:
                line_to_write = ','.join(line)
                f_output.write(line_to_write+"\n")
                connectivity[(str(line[1]))] = [str(line[1])]
        else:
            f_output.write(','.join(line)+"\n")
        number_of_line = number_of_line + 1
 
    print(connectivity)    

    
    if algorithm != '-':
        logging.info('Generating influence net...')   
     
        infile.seek(0)
        for line in infile:
            for tweet in ensure_flattened(json.loads(line)):
                if 'referenced_tweets' in tweet:
                    for x in tweet['referenced_tweets']:
                        if 'retweeted' in x['type']:
                            author_name = x['author']['username']
                            retweeter = tweet['author']['username']
                            is_allowed = True
                            if is_interval:
                                created_at_time = pd.to_datetime(created_at, utc=True)
                                if not start_time <= created_at_time <= end_time:
                                    is_allowed = False
                            if is_allowed:
                                if author_name in connectivity:
                                    if str(retweeter) not in connectivity[author_name]:
                                        connectivity[author_name].append(str(retweeter))
        print(connectivity)
    
        logging.info('Computing distance matrix')
        elite_authors=np.empty(len(connectivity),dtype="<U10")
    
        n_matrix = np.zeros((len(connectivity),len(connectivity)))
        i=0
        for element1 in connectivity:
            j=0
            elite_authors[i]= element1
            for element2 in connectivity:
                n_matrix[i,j]=len(set(connectivity[element1]) & set(connectivity[element2]))
                j=j+1
            i=i+1
    
        #print(n_matrix)
    
        phi = np.zeros((len(connectivity),len(connectivity)))
        
        for i in range(len(connectivity)):
            for j in range(len(connectivity)):
                aux = 0
                for k in range(len(connectivity)):
                    aux += n_matrix[i,k]*n_matrix[j,k]*n_matrix[k,i]*n_matrix[k,j]
                aux = max(aux, 0.000001)
                phi[i,j] = (n_matrix[i,i]*n_matrix[j,j]-n_matrix[i,j]*n_matrix[j,i])/np.sqrt(aux)
    
        #phi[np.isinf(phi)] = 10000
        #print(phi)
        f_temp_input.close()
        f_output.close()
    
        logging.info('Finished.')
        os.remove(second_temporal_csv)
        y = ssd.squareform(phi)
    
        print(y, method, algorithm)
    
    
        logging.info('Computing clusters...')
        
        plt.figure()
        if algorithm == 'nn_chain':
            Z, pol = cl.agglomerative_clustering(y, method='ward', alpha=1, K=None, verbose=0, algorithm='nn_chain')
            print (Z, pol)
            dendro = sch.dendrogram(Z, labels = elite_authors)
        elif algorithm == 'generic':
            if method == 'centroid':
                Z, pol = cl.agglomerative_clustering(y, method='centroid', alpha=1, K=None, verbose=0, algorithm='generic')
                print (Z, pol)
                dendro = sch.dendrogram(Z, labels = elite_authors)
            elif method == 'poldist':
                Z, pol = cl.agglomerative_clustering(y, method='poldist', alpha=1, K=None, verbose=0, algorithm='generic')
                print (Z, pol)
                dendro = sch.dendrogram(Z, labels = elite_authors)
            elif method =='ward':
                Z, pol = cl.agglomerative_clustering(y) 
                print (Z, pol)
                dendro = sch.dendrogram(Z, labels = elite_authors)
    
        plt.savefig('plt.png', format='png', bbox_inches='tight')

if __name__ == '__main__':
    main()
