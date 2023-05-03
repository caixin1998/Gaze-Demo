#generate train and test split for the dataset
import json
import os
import random
dataset_path = "/home1/caixin/GazeData/VIPLGaze538/data"
person_list = []
for file in os.listdir(dataset_path):
    person = file.split(".")[0]
    person_list.append(person)
random.shuffle(person_list)
train_list = person_list[:18]
test_list = person_list[18:]
with open(os.path.join(dataset_path,"train_valid_split.json"), "w") as f:
    json.dump({"train": train_list, "valid": test_list}, f)