import pandas as pd
import numpy as np
import os

# Get path
path = os.getcwd() + "\code\data\games.csv"
data = pd.read_csv(path)

# Select columns for X
champ_columns = ["t1_champ1id", "t1_champ2id", "t1_champ3id", "t1_champ4id", "t1_champ5id",
            "t2_champ1id", "t2_champ2id", "t2_champ3id", "t2_champ4id", "t2_champ5id"]

x_raw = data[champ_columns]

# One hot encode
one_hot = pd.get_dummies(x_raw, columns=champ_columns)

# Reshape and move axi to correct order
X_one_hot = one_hot.values.reshape((51490, 138, 2, 5))
X_one_hot = np.moveaxis(X_one_hot, [2, 3], [3, 2])

# Save X
np.save("X", X_one_hot)

# Select column
y = data[["winner"]]

# Change to 0/1
Y = y.values - 1

# Save Y
np.save("Y", Y)
