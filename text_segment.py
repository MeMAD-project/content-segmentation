import os
import time
import json
import pysrt
import pickle
import argparse

import pandas as pd
import numpy as np

from sentence_transformers import models, SentenceTransformer, util

parser = argparse.ArgumentParser(description='Textual Content Segmentation')

parser.add_argument("-s", "--subtitles_path", type=str, help="Path to the subtitles path")
parser.add_argument("-o", "--output_path", type=str, help="Path to save results.", default="./")
parser.add_argument("-w", "--window_size", type=int, help="Neighborhood size for similarity.", default=3)
parser.add_argument("-am", "--aggregation_method", help="", choices=['average', 'product'], default='average')
parser.add_argument("-sm", "--scoring_method", help="", choices=['minima', 'lowest'], default='minima')

args = parser.parse_args()

assert(args.subtitles_path.endswith('srt'))

def to_seconds(str_time):
    if '.' not in str_time: 
        str_time = str_time + '.000'
    assert(time.strptime(str_time,'%H:%M:%S.%f'))
    seconds = sum(x * float(t) for x, t in zip([3600, 60, 1], str_time.split(":")))
    return seconds

def process_program(sim_matrix, subtitle_end_times, window_size, aggregation_method, scoring_method):
    N = sim_matrix.shape[0]
    scores_dic = {'average':[], 'product':[]}
    
    for i in range(N):
        neighbors = range(i, min(i+1+window_size, N))
        neighbors_scores = [sim_matrix[i][j] for j in neighbors]
        scores_dic['average'].append(np.mean(neighbors_scores))
        scores_dic['product'].append(np.product(neighbors_scores))
    
    scores = scores_dic[aggregation_method]

    if scoring_method == 'minima':
        minima = []
        for i in range(1, N-1):
            if scores[i] <= scores[i - 1] and scores[i] <= scores[i + 1]:
                minima.append((i, scores[i]))
        sorted_scores = [(i, score) for i, score in sorted(minima, key=lambda x: x[1])]
    elif scoring_method == 'lowest':
        sorted_scores = [(i, score) for i, score in sorted(enumerate(scores), key=lambda x: x[1])]
    else:
        raise Exception('Scoring method "' + aggregation_method + '" not supported!')
    
    boundaries = [(subtitle_end_times[i], s) for i, s in sorted_scores]
    
    return boundaries


data = []
print('Processing file', args.subtitles_path, '..')
subs =  pysrt.open(args.subtitles_path)
for s in subs:
    sub = {'content': s.text}
    sub['start'] = str(s.start).replace(',', '.')
    sub['start_s'] = to_seconds(str(s.start).replace(',', '.'))
    sub['end'] = str(s.end).replace(',', '.')
    sub['end_s'] = to_seconds(str(s.end).replace(',', '.'))
    sub['duration_s'] = sub['end_s'] - sub['start_s']
    data.append(sub)
print(len(data), "subtitles.")

df = pd.DataFrame(data)[['content', 'start', 'end', 'start_s', 'end_s', 'duration_s']]

print('Loading multilingual distiluse SBERT..', end='')
sbert = SentenceTransformer('distiluse-base-multilingual-cased') 
print(' done.')

print('Embedding the subtitles..')
embeddings = sbert.encode(df.content.values.tolist(), convert_to_tensor=True, show_progress_bar=True)

res = util.pytorch_cos_sim(embeddings, embeddings).numpy()

times = process_program(sim_matrix = res, 
                        subtitle_end_times = df.end.values, 
                        window_size = args.window_size, 
                        aggregation_method = args.aggregation_method, 
                        scoring_method = args.scoring_method)

output_filename = os.path.join(args.output_path, os.path.basename(args.subtitles_path).split('.')[0])

print('Segmentation candidates saved at', output_filename + '.csv')

results = pd.DataFrame(times, columns=['times', 'scores'])
results.to_csv(output_filename + '.csv', index=False)

data_dict = {'start': df.start_s.values.tolist(),
             'end': df.end_s.values.tolist(), 
             'similarity': res}

pickle.dump(data_dict, open(output_filename + '.pickle', 'wb'))

print('Segmentation data and similarity scores saved at', output_filename + '.pickle')