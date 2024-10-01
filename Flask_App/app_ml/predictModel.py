import joblib
import os
import numpy as np
import tensorflow as tf


def predictModel(algorithm, df):
    model = None
    scaler = None
    ohe = None
    threshold = None

    current_path = os.path.dirname(os.path.abspath(__file__))
    print(current_path)
    path = os.path.join(current_path, "models")
    


    if (algorithm == "MLP"):
        model = tf.keras.models.load_model(path+"/mlp_model.h5")
        threshold = joblib.load(path + "/threshold.pkl")

    elif (algorithm == "KNN"):
        model = joblib.load(path+"/knn_model.pkl")

    elif (algorithm == "Logistic Regression"):
        model = joblib.load(path + "/log_model.pkl")

    scaler = joblib.load(path+"/scaler.pkl")
    ohe = joblib.load(path+"/ohe.pkl")

    if (model == None):
        return -1

    categorical = df.select_dtypes(include=['object']).columns
    numerical = df.select_dtypes(exclude=['object']).columns

    data = None

    encoded_cols = ohe.transform(df[categorical])
    df = df.drop(categorical, axis=1)

    data = np.concatenate([scaler.transform(df), encoded_cols], axis=1)

    result = model.predict(data)
    if (algorithm == "MLP"):
        result = (result >= threshold).astype(int)

    return result
