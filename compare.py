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
def make_bar_graph():
    plt.style.use('seaborn-v0_8-whitegrid')

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
    print(plt.style.available)
    plt.show()

def make_dataframe():
    data = {'10 PR': pr10, '20 PR': pr20, '30 PR': pr30, '40 PR': pr40, '50 PR': pr50}
    df = pd.DataFrame(data)
    df.index = ['10ep-1iter', '10ep-8iter', '10ep-16iter', '25ep-8iter', '25ep-16iter', '50ep-8iter']
    print(df)

def compare_base_and_pruned(batch_name, pruned_name, exp_results_path, base_results_path):
    """
    Params:
        (string) batch_name -> ex. "batch-1 (10ep-1iter)"
        (string) pruned_name -> ex. "50ep-30pr-8iter"
        (string) exp_results_path -> filepath of the json file where the experiment results are stored
        (string) base_results_path -> filepath of the json file where the base model results are stored
    """

    data = {}
    json_data = {}
    with open(exp_results_path, 'r') as f:
        json_data = json.load(f)

    if batch_name not in json_data or pruned_name not in json_data[batch_name]:
        print('ERROR: Either batch_name not in exp_results_path or pruned_name not in specified batch')
        return    

    data[pruned_name] = json_data[batch_name][pruned_name]

    with open(base_results_path, 'r') as f:
        json_data = json.load(f)

    for key, value in json_data.items():
        data[key] = value

    models = list(data.keys())
    params = []
    accuracy = []
    speed = []

    for key, value in data.items():
        idx = 0
        if key == "v8small-50ep-16bs":
            idx = 1
        if key == pruned_name:
            idx = 2

        params.insert(idx, value['no_of_params'])
        accuracy.insert(idx, value['map50_95'])
        speed.insert(idx, value['inference_speed'])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(models, params, color='red', label='No. of params')
    ax.barh(models, accuracy, color='blue', label='Accuracy (%)')
    ax.barh(models, speed, color='green', label='Speed (ms)')

    # Set labels and title
    ax.set_xlabel('Normalized Value')
    ax.set_title('Model Comparison')
    ax.legend()

    plt.show()


if __name__ == '__main__':
    compare_base_and_pruned('batch-6 (50ep-8iter)', '50ep-40pr-8iter', 'experiment_results.json', 'base_model_results.json')