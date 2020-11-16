import pandas as pd
import numpy as np
DATA_DIR = "data"

recommandation_df = pd.read_csv('{}/normalizedata.csv'.format(DATA_DIR)).sort_values(by=['time']).values

test = np.array([[1,2,4], [1,2,4]])

print(test[:,2][0])
