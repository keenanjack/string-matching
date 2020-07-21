import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix
from utils import progressBar

def process(leftcsv, rightcsv, similiarity, ntop):
  print(f'Reading left CSV...')
  left_names = pd.read_csv(leftcsv)
  print(f'Reading right CSV...')
  right_names = pd.read_csv(rightcsv)
  all_names = left_names.append(right_names).reset_index()
  names = all_names['Name']
  vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
  print(f'Fitting vectorizer...')
  tf_idf_matrix = vectorizer.fit_transform(names)
  print(f'Generating matches...')
  matches = scipy_cossim_top(tf_idf_matrix, tf_idf_matrix.transpose(), ntop, similiarity)
  print(f'Generating df...')
  result = get_matches_df(matches, names, top=None)
  non_exact_results = result.loc[(result['actual_name'].isin(right_names['Name']) & (result['actual_name'] != result['likely_name']) & (result['similairity'] >= similiarity))].drop_duplicates()
  non_exact_results_rejoined = pd.merge(non_exact_results, left_names, left_on='likely_name', right_on='Name', how='left')
  print(f'Storing df...')
  return non_exact_results_rejoined.to_csv('data/output.csv')

def get_csr_ntop_idx_data(csr_row, ntop):
    """
    Get list (row index, score) of the n top matches
    """
    nnz = csr_row.getnnz()
    if nnz == 0:
        return None
    elif nnz <= ntop:
        result = zip(csr_row.indices, csr_row.data)
    else:
        arg_idx = np.argpartition(csr_row.data, -ntop)[-ntop:]
        result = zip(csr_row.indices[arg_idx], csr_row.data[arg_idx])

    return sorted(result, key=lambda x: -x[1])

def scipy_cossim_top(A, B, ntop, lower_bound=0):
    C = A.dot(B)
    return [get_csr_ntop_idx_data(row, ntop) for row in C]

def ngrams(string, n=3):
    string = re.sub(r'[,-./]',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


def get_matches_df(sparse_matrix, name_vector, top=100):

    if top:
        nr_matches = top
    else:
        nr_matches = sum([len(listElem) for listElem in sparse_matrix if listElem])

    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)


    i = 0
    for index, match in enumerate(sparse_matrix):
        if match:
            for entry in match:
                left_side[i] = name_vector[index]
                right_side[i] = name_vector[entry[0]]
                similairity[i] = entry[1]
                i += 1
        else:
            next
    return pd.DataFrame({'actual_name': left_side,'likely_name': right_side,'similairity': similairity})