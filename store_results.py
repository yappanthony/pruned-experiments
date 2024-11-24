from ultralytics import YOLO
import torch
from torch import nn
from ultralytics.nn.modules import Conv, Bottleneck
import os
import json

# This class is required to be present in the file where you use the pruned weights
class C2f_v2(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # y = list(self.cv1(x).chunk(2, 1))
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
    
def store_results():
    """
    Validates YOLO weights on a given dataset
    
    Stores particular metrics in a json file

    Metrics to store: 
    precison, recall, mean precision, mean recall, map50, map50-95, all classes' map50-95, number of model parameters
    """
    results_dict = {}

    batch_experiments_folders = ['batch-1 (10ep-1iter)', 'batch-2 (10ep-8iter)', 'batch-3 (10ep-16iter)', 'batch-4 (25ep-8iter)']
    for batch in batch_experiments_folders:
        folders = os.listdir(batch)
        for folder in folders:
            model = YOLO(f"{batch}/{folder}/weights/best.pt")
            no_of_params = sum(p.numel() for p in model.parameters())
            results = model.val()

            if str(batch) not in results_dict:
                results_dict[str(batch)] = {}
           
            results_dict[str(batch)][str(folder)] = {
                'precision': {
                    "car_front": round(results.box.p[0], 3),
                    "car_back:": round(results.box.p[1], 3),
                    "license_plate": round(results.box.p[2], 3)
                },
                'mean_precision': round(results.box.mp, 3),
                'recall': {
                    "car_front": round(results.box.r[0], 3),
                    "car_back:": round(results.box.r[1], 3),
                    "license_plate": round(results.box.r[2], 3)
                },
                'mean_recall': round(results.box.mr, 3),
                'map50': round(results.box.map50, 3),
                'map50_95': round(results.box.map, 3),
                'all_map50_95': {
                    "car_front": round(results.box.maps[0], 3),
                    "car_back:": round(results.box.maps[1], 3),
                    "license_plate": round(results.box.maps[2], 3)
                },
                'no_of_params': no_of_params
            }

    with open('experiment_results.json', 'w') as outfile:
        json.dump(results_dict, outfile, indent=4)


if __name__ == "__main__":
    store_results()
