import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix
from utils import progressBar

def process(leftcsv, rightcsv, similiarity):
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
  matches = scipy_cossim_top(tf_idf_matrix, tf_idf_matrix.transpose(), 10, similiarity)
  print(f'Generating df...')
  result = get_matches_df(matches, names, top=None)
  print(result.loc[(result['left_side'].isin(right_names['Name']) & (result['left_side'] != result['right_side']))].drop_duplicates())
  return result.loc[(result['left_side'].isin(right_names['Name']) & (result['left_side'] != result['right_side']))].drop_duplicates()

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
    return C

def ngrams(string, n=3):
    string = re.sub(r'[,-./]',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


def get_matches_df(sparse_matrix, name_vector, top=100):
    non_zeros = sparse_matrix.nonzero()

    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    if top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size

    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)

    for index in progressBar(range(0, nr_matches)):
        left_side[index] = name_vector[sparserows[index]]
        right_side[index] = name_vector[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]

    return pd.DataFrame({'left_side': left_side,
                          'right_side': right_side,
                           'similairity': similairity})