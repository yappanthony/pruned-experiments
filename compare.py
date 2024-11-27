import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

# batch-1, batch-2, batch-3, batch-4
experiment_batches = [] 

# Store the map50-95 here
pr10 = []
pr20 = []
pr30 = []
pr40 = []
pr50 = []

data = {}
with open('experiment_results.json', 'r') as f:
    data = json.load(f)

for key, value in data.items():
    experiment_batches.append(key)
    # experiment_batches will contain 4 elements
    for key2, value2 in value.items():
        i = int(key.split()[0][-1]) - 1 # "batch-1 (10ep-1iter)" -> split()[0] = batch-1 -> [-1] = 1

        if "10pr" in key2:
            pr10.insert(i, value2['map50_95'])
        elif "20pr" in key2:
            pr20.insert(i, value2['map50_95'])
        elif "30pr" in key2:
            pr30.insert(i, value2['map50_95'])
        elif "40pr" in key2:
            pr40.insert(i, value2['map50_95'])
        elif "50pr" in key2:
            pr50.insert(i, value2['map50_95'])

# This should create a table containing mAP50-95 values
# The columns are the batches from batch 1-4 
# The rows are the prune rates from 10-50%
print(pr10)
print(pr20)
print(pr30)
print(pr40)
print(pr50)

# Create a bar chart
def bar_graph():
    x = len(experiment_batches)
    width = 0.6 / x

    bar_positions = np.arange(len(experiment_batches)) - (x - 1) * width / 2

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(bar_positions, pr10, width, label='10 PR')
    rects2 = ax.bar(bar_positions + width, pr20, width, label='20 PR')
    rects3 = ax.bar(bar_positions + 2 * width, pr30, width, label='30 PR')
    rects4 = ax.bar(bar_positions + 3 * width, pr40, width, label='40 PR')
    rects5 = ax.bar(bar_positions + 4 * width, pr50, width, label='50 PR')

    ax.set_xticks(bar_positions + width)
    ax.set_xticklabels(experiment_batches)

    ax.set_title('Model Performance Comparison')
    ax.set_ylabel('mAP50-95')

    ax.legend()
    plt.ylim(0.6, 0.9)
    plt.show()

def make_dataframe():
    data = {'10 PR': pr10, '20 PR': pr20, '30 PR': pr30, '40 PR': pr40, '50 PR': pr50}
    df = pd.DataFrame(data)
    df.index = ['10ep-1iter', '10ep-8iter', '10ep-16iter', '25ep-8iter', '25ep-16iter', '50ep-8iter']
    print(df)


if __name__ == '__main__':
    make_dataframe()