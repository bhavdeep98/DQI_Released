import os
import pandas as pd
import csv

with open("/scratch/user/traingood.csv") as f:
    test_bad = csv.DictReader(f, delimiter=',', restkey='rest')
    test_bad_df = pd.DataFrame(test_bad)

new_testBad = test_bad_df[['gold_label','sentence1','sentence2']].copy()
new_testBad = new_testBad.mask(new_testBad.eq('None')).dropna()
new_testBad.to_csv("/scratch/user/newTrainGood",sep="\t", header=False, index=False)
