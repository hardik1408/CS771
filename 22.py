import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import joblib



####################################################
#################### DATASET 1 #####################
####################################################

###### Loading and preprocessing the test data #######

# Load the CSV files
train_df = pd.read_csv('datasets/train/train_emoticon.csv')
validation_df = pd.read_csv('datasets/valid/valid_emoticon.csv')
test_df = pd.read_csv('datasets/test/test_emoticon.csv')

# Assuming the CSV files have 'emojis' and 'label' columns
train_texts = train_df['input_emoticon'].values
train_labels = train_df['label'].values

validation_texts = validation_df['input_emoticon'].values
validation_labels = validation_df['label'].values

test_texts = test_df['input_emoticon'].values

total_train = [] # A list of all the emoticons
for i in train_texts:
    total_train += i

emoji_freq = dict() # Each emoticon mapped to its frequency of occurence
for i in total_train:
    if i in emoji_freq:
        emoji_freq[i] += 1
    else:
        emoji_freq[i] = 1

emoji_0_1 = dict()

for i in emoji_freq:
    emoji_0_1[i] = (0,0)

for i in emoji_freq:
    for j in range(len(train_texts)):
        if i in train_texts[j]:
            if train_labels[j] == 1:
                emoji_0_1[i] = (emoji_0_1[i][0], emoji_0_1[i][1]+1)
            else:
                emoji_0_1[i] = (emoji_0_1[i][0]+1, emoji_0_1[i][1])

most_frequent_emojis = []
for i in emoji_0_1:
    if emoji_0_1[i] == (3576, 3504):
        most_frequent_emojis.append(i)

test_texts_mod = []
for i in test_texts:
    s = ''
    for j in i:
        if j not in most_frequent_emojis:
            s += j
    test_texts_mod.append(s)

test_texts_mod = np.array(test_texts_mod, dtype=object)

#  Tokenize and pad the emoji sequences
tokenizer = Tokenizer(char_level=True)  # Tokenizing each emoji as a character
tokenizer.fit_on_texts(train_texts)  # Fit only on training data

# Convert texts to sequences
test_sequences = tokenizer.texts_to_sequences(test_texts_mod)

# Pad the sequences to the same length
max_len = max([len(seq) for seq in test_sequences])  # Maximum sequence length in train data
test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post')


####### Loading the model #######

# Load the entire model (architecture + weights + optimizer)
emoticon_model = load_model('./models/dataset1.keras')

predictions = emoticon_model.predict(test_padded)
predicted_labels = (predictions > 0.5).astype(int)
predicted_labels = [i[0] for i in predicted_labels]

def save_predictions_to_file(predictions, filename):
    with open(filename, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

# saving prediction to text files
save_predictions_to_file(predicted_labels, "pred_emoticon.txt")



####################################################
#################### DATASET 2 #####################
####################################################

path = './datasets/train/train_feature.npz'
with np.load(path) as data:
    train_examples = data['features']
    train_labels = data['label']

path = './datasets/valid/valid_feature.npz'
with np.load(path) as data:
    valid_examples = data['features']
    valid_labels = data['label']

path = './datasets/test/test_feature.npz'
with np.load(path) as data:
    test_examples = data['features']

train_examples = np.asarray(train_examples, dtype=np.float32)
train_labels = np.asarray(train_labels, dtype=np.int32)
valid_examples = np.asarray(valid_examples, dtype=np.float32)
valid_labels = np.asarray(valid_labels, dtype=np.int32)
test_examples = np.asarray(test_examples, dtype=np.float32)


train_examples_flat = train_examples.reshape(train_examples.shape[0], -1)
valid_examples_flat = valid_examples.reshape(valid_examples.shape[0], -1)
test_examples_flat = test_examples.reshape(test_examples.shape[0], -1)

clf = joblib.load('./models/dataset2.joblib')

predicted_labels = clf.predict(test_examples_flat)

def save_predictions_to_file(predictions, filename):
    with open(filename, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

# saving prediction to text files
save_predictions_to_file(predicted_labels, "pred_deepfeat.txt")


####################################################
#################### DATASET 3 #####################
####################################################


df_test = pd.read_csv('./datasets/test/test_text_seq.csv')
# Extract input sequences and labels
X_test = df_test['input_str'].values

label_encoder = LabelEncoder()
label_encoder.fit(list('0123456789'))

# Convert input_str into sequences of encoded digits, removing the first three zeroes
def encode_sequence(sequence):
    return label_encoder.transform(list(sequence.lstrip('0')))  # Remove leading zeroes

X_test_encoded = [encode_sequence(seq) for seq in X_test]

# Pad the sequences to ensure they all have the same length (47 in this case)
X_test_padded = pad_sequences(X_test_encoded, maxlen=47, padding='post')


d3_model = load_model('./models/dataset3.keras')

y_pred = d3_model.predict(X_test_padded)
predicted_labels = np.argmax(y_pred, axis=1)

def save_predictions_to_file(predictions, filename):
    with open(filename, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

# saving prediction to text files
save_predictions_to_file(predicted_labels, "pred_textseq.txt")


####################################################
##################### COMBINED #####################
####################################################

df_1 = pd.read_csv('./datasets/test/test_emoticon.csv')
df_3 = pd.read_csv('./datasets/test/test_text_seq.csv')

path = './datasets/test/test_feature.npz'
with np.load(path) as data:
  test_examples = data['features']

test_examples = np.asarray(test_examples, dtype=np.float32)

test_examples_flat = test_examples.reshape(test_examples.shape[0], -1)
df_2 = pd.DataFrame(test_examples_flat) 

df_1_no_label = df_1
df_2_no_label = df_2

# Merged testing data
merged_df = pd.concat([df_1_no_label, df_2_no_label, df_3], axis=1)

test_texts = merged_df['input_emoticon'].values

tokenizer = Tokenizer(char_level=True)  
tokenizer.fit_on_texts(test_texts) 
test_sequences = tokenizer.texts_to_sequences(test_texts)

max_len = max([len(seq) for seq in test_sequences])   
test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post')

merged_df.drop(columns=['input_emoticon'], inplace=True)
test_padded_df = pd.DataFrame(test_padded, columns=[f'padded_{i}' for i in range(test_padded.shape[1])])
merged_df = pd.concat([test_padded_df,merged_df], axis=1)

def split_into_chunks(input_str, chunk_size=2):
    return [input_str[i:i+chunk_size] for i in range(0, len(input_str), chunk_size)]

chunks_df = merged_df['input_str'].apply(split_into_chunks).apply(pd.Series)
chunks_df.columns = [f'chunk_{i+1}' for i in range(chunks_df.shape[1])]
merged_df = pd.concat([merged_df, chunks_df], axis=1)

merged_df.drop(columns=['input_str'], inplace=True)

clf = joblib.load('./models/combined.joblib')

test = merged_df

# Convert column names to strings
test.columns = test.columns.astype(str)

y_pred = clf.predict(test)

def save_predictions_to_file(predictions, filename):
    with open(filename, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

# saving prediction to text files
save_predictions_to_file(y_pred, "pred_combined.txt")