from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

cur_dir = os.getcwd()
customers = pd.read_csv(cur_dir + "/german_credit_data.csv")
customers['Risk'] = customers['Risk'].replace("good", 0)
customers['Risk'] = customers['Risk'].replace("bad", 1)
customers = customers.drop("Unnamed: 0", axis='columns')
customers_train, customers_test = train_test_split(customers, test_size= 0.2, random_state=42, stratify=customers['Risk'])
customers_train.to_csv(cur_dir + '/train.csv', index = False)
customers_test.to_csv(cur_dir + '/test.csv', index=False)
