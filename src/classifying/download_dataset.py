import kagglehub
import os
import shutil
import pandas as pd

if __name__ == "__main__":
    # Download latest version
    path = kagglehub.dataset_download(
        "ahemateja19bec1025/traffic-sign-dataset-classification"
    )

    print("Path to dataset files:", path)

    # Move files from path to data/
    destination = "data/"
    if not os.path.exists(destination):
        os.makedirs(destination)

    for filename in os.listdir(path):
        shutil.move(os.path.join(path, filename), os.path.join(destination, filename))
    shutil.rmtree(path)

    ## código usado por si fuera necesario renombrar carpetas

    num_to_label = pd.read_csv("data/labels.csv", index_col="ClassId").to_dict()["Name"]
    num_to_label = {str(k): v for k, v in num_to_label.items()}
    num_to_label = {str(k): v.replace("/", "") for k, v in num_to_label.items()}
    print(num_to_label)

    train_data_dir = "data/traffic_Data/DATA/"
    for dir in os.listdir(train_data_dir):
        if dir in num_to_label.keys():
            dir_path = os.path.join(train_data_dir, dir)
            new_dir_path = os.path.join(train_data_dir, num_to_label[dir])
            print(f"Renaming {dir_path} to {new_dir_path}")
            if os.path.exists(dir_path):
                if os.path.exists(new_dir_path):
                    shutil.move(dir_path, new_dir_path)
                else:
                    os.renames(dir_path, new_dir_path)
            else:
                print(f"Path {dir_path} does not exist")

    test_data_dir = "data/traffic_Data/TEST/"

    for image in os.listdir(test_data_dir):
        num = image[:3]
        try:
            num = str(int(num))
            label = num_to_label[num]
            dir_path = os.path.join(test_data_dir, label)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            image_path = os.path.join(test_data_dir, image)
            shutil.move(image_path, dir_path)
        except:
            pass
