import pandas as pd


if __name__ == '__main__':
    dataset_csv_pathes = ["data/Competition_data/train.csv", "data/Fake_data_unet_mask_copy/train_f.csv",
                          "data/Fake_data_simple_mask_copy/train_f.csv"]
    labels = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
                   'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
                   'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
                   'Swan Ganz Catheter Present']
    for path in dataset_csv_pathes:
        df = pd.read_csv(path)
        labels_count_dict = {}
        for label in labels:
            loc = df.loc[df[label] == 1].index
            labels_count_dict[label] = len(loc)
        print("for file: " + path + " the label counts are:")
        print(labels_count_dict)
