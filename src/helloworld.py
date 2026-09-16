import sys
import torch
import pandas as pd
import sklearn

print("Python exe:", sys.executable)
print("Torch ver:", torch.__version__)
print("Pandas ver:", pandas.__version__)
print("SK-Learn ver:", sklearn.__version__)


# TODO: Delete later
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir, 'realestate.csv'))
