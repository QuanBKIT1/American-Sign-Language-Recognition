from sklearn.model_selection import train_test_split
import pandas as pd
import configs as cf

def split_meta_data(meta_data,test_size):
    X_1, X_2, y_1, y_2 = train_test_split(
    meta_data["image_path"], meta_data["label"],
    test_size=test_size,
    random_state=2023,
    shuffle=True,
    stratify=meta_data["label"]
    )
    
    data_1 = pd.DataFrame({
    "image_path": X_1,
    "label": y_1
    })
    
    data_2 = pd.DataFrame({
        "image_path": X_2,
        "label": y_2
    })
    return data_1, data_2