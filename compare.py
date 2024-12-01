import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

batch_experiments = [
        '10ep-1iter', 
        '10ep-4iter',
        '10ep-8iter', 
        '10ep-12iter',
        '10ep-16iter', 
        # '25ep-1iter', 
        # '25ep-4iter',
        # '25ep-8iter',
        # '25ep-12iter',
        # '25ep-16iter',
        # '50ep-8iter',
        # '50ep-12iter'
    ]

def get_metric(metric):
    """
    Param:
    metric (string) 

    Returns 5 lists (one for each prune rate) containing the specified metric
    """

    pr10 = []
    pr20 = []
    pr30 = []
    pr40 = []
    pr50 = []

    data = {}
    with open('experiment_results.json', 'r') as f:
        data = json.load(f)

    for key, value in data.items():
        for key2, value2 in value.items():
            i = int(key.split()[0].split('-')[-1]) - 1

            if "10pr" in key2:
                pr10.insert(i, value2[metric])
            elif "20pr" in key2:
                pr20.insert(i, value2[metric])
            elif "30pr" in key2:
                pr30.insert(i, value2[metric])
            elif "40pr" in key2:
                pr40.insert(i, value2[metric])
            elif "50pr" in key2:
                pr50.insert(i, value2[metric])

    return pr10, pr20, pr30, pr40, pr50

def make_dataframe():
    pr10, pr20, pr30, pr40, pr50 = get_metric('no_of_params')

    data = {'10 PR': pr10, '20 PR': pr20, '30 PR': pr30, '40 PR': pr40, '50 PR': pr50}
    df = pd.DataFrame(data)
    df.index = batch_experiments
    df.to_csv('map95_results.csv')


if __name__ == '__main__':
    make_dataframe()