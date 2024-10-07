import numpy as np
import pandas as pd


# these are dummy models
class MLModel():
    def __init__(self) -> None:
        pass
    
    def train(self, X, y):
        NotImplemented
    
    def predict(self, X):
        NotImplemented
    
class TextSeqModel(MLModel):
    def __init__(self) -> None:
        pass

    def predict(self, X):# random predictions
        return np.random.randint(0,2,(len(X)))
    
    
class EmoticonModel(MLModel):
    def __init__(self) -> None:
        pass

    def predict(self, X):# random predictions
        return np.random.randint(0,2,(len(X)))
    
class FeatureModel(MLModel):
    def __init__(self) -> None:
        pass

    def predict(self, X): # random predictions
        return np.random.randint(0,2,(len(X)))
    
class CombinedModel(MLModel):
    def __init__(self) -> None:
        pass

    def predict(self, X1, X2, X3): # random predictions
        return np.random.randint(0,2,(len(X1)))
    
    
def save_predictions_to_file(predictions, filename):
    with open(filename, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

if __name__ == '__main__':
    # read datasets
    test_feat_X = np.load("datasets/test/test_feature.npz", allow_pickle=True)['features']
    test_emoticon_X = pd.read_csv("datasets/test/test_emoticon.csv")['input_emoticon'].tolist()
    test_seq_X = pd.read_csv("datasets/test/test_text_seq.csv")['input_str'].tolist()
    
    # your trained models 
    feature_model = FeatureModel()
    text_model = TextSeqModel()
    emoticon_model  = EmoticonModel()
    best_model = CombinedModel()
    
    # predictions from your trained models
    pred_feat = feature_model.predict(test_feat_X)
    pred_emoticons = emoticon_model.predict(test_emoticon_X)
    pred_text = text_model.predict(test_seq_X)
    pred_combined = best_model.predict(test_feat_X, test_emoticon_X, test_seq_X)
    
    # saving prediction to text files
    save_predictions_to_file(pred_feat, "pred_feat.txt")
    save_predictions_to_file(pred_emoticons, "pred_emoticon.txt")
    save_predictions_to_file(pred_text, "pred_text.txt")
    save_predictions_to_file(pred_combined, "pred_combined.txt")
    
    
