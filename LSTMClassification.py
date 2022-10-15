
# ## Import Libraries
# Loading all libraries to be used
import copy
import numpy as np
import re
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import pad_sequences
from keras.callbacks import ModelCheckpoint
# fix random seed for reproducibility
np.random.seed(7)
# from prettytable import PrettyTable
import warnings
warnings.filterwarnings('ignore')
nltk.download('stopwords')

candidate_Labels={"Politics":["Donald Trump","Politics & Policy","Hillary Clinton","Congress","Politics","Joe Biden","Bernie Sanders","Ted Cruz"],
                  "Business":["Business & Finance"],
                  "Opinion":["World","Debates"],
                  "Tech":["Technology","On Instagram","Television","Telecoms"  ],
                  "Science":["Space","Science & Health","Neuroscience","Climate Change"],
                  "Health":["Health Care","Infectious Disease"],
                  "Sports":["Sports","2016 Rio Olympics"],
                  "Entertainment":["Movies" ,"Comic Books","Music","Energy & Environment","Game of Thrones","Episode of the Week","Star Wars","Avengers: Age of Ultron","Game of Thrones, season 6, episode 10","Mad Men, season 7, episode 8","Mad Men, season 7, episode 11"],
                  "Education":["Books","Education"],
                  "Fashion":["Culture","Religion" ],
                  "Food":[],
                  "Travel":["Transportation"],
                  "Real Estate":["Small Business"],
                  "Legal":["Supreme Court ","Policy","LGBTQ","Marriage Equality"],
                  "Crime":["Criminal Justice","Gun Violence","Gender-Based Violence","Hate Crimes"]}



def readData(file):
    titles = []
    categories = []
    links=[]
    with open(file,'r') as tsv:
        count = 0;
        for line in tsv:
            a = line.strip().split('\t')[:6]
            # if a[2] in ['Business & Finance', 'Health Care', 'Science & Health', 'Politics & Policy', 'Criminal Justice']:
            title = a[0].lower()
            title = re.sub('\s\W', ' ', title)
            title = re.sub('\W\s', ' ', title)
            for k, v in candidate_Labels.items():
                if a[2] in v:
                    categories.append(k)
                    titles.append(title)
                    links.append(a[5])
    return titles[1:],categories[1:],links[1:]

def buildDictionary(textlist,minrepition=10):
    wordslist=[text.lower().split() for text in textlist]
    uniquewordscount=list(map(lambda x:x[0], filter(lambda x:x[1]>minrepition,zip(*np.unique(sum(wordslist,[]),return_counts=True)))))
    wordtonumber={w:i+2 for i,w in enumerate(uniquewordscount)}
    wordslist=[[wordtonumber.get(word,1) for word in words] for words in wordslist]
    return wordslist,wordtonumber

# Dataset: https://data.world/elenadata/vox-articles
titles,categories,links=readData('../Files/dsjVoxArticles.tsv')
wordslist,wordtonumber=buildDictionary(titles)
nb_classes=np.unique(categories).shape[0]
categoriestonumber={cat:i for i,cat in enumerate(np.unique(categories))}
categoriesonehot=list(map(lambda x:tf.one_hot(categoriestonumber[x], nb_classes),categories))
n_uniquewords=len(wordtonumber)
embedding_dim=64

print(len(wordslist),len(categoriesonehot))
X_train,X_test,y_train,y_test = train_test_split(wordslist,categoriesonehot,test_size = 0.2)
print("Shape of train data:", len(X_train))
print("Shape of Test data:", len(X_test))


max_title_length = 25
X_train = pad_sequences(X_train, maxlen=max_title_length)
X_test = pad_sequences(X_test, maxlen=max_title_length)

batchsize=64
train_ds=tf.data.Dataset.from_tensor_slices((X_train,y_train))
test_ds=tf.data.Dataset.from_tensor_slices((X_test,y_test))
train_ds = train_ds.batch(batchsize).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.batch(batchsize).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

model = tf.keras.Sequential([
  layers.Embedding(n_uniquewords, embedding_dim,input_length=max_title_length),
  layers.LSTM(1024,return_sequences=True),
  layers.Dropout(0.2),
  layers.LSTM(512,return_sequences=True),
  layers.Dropout(0.2),
  layers.LSTM(256,return_sequences=True),
  layers.Dropout(0.2),
  layers.LSTM(128,return_sequences=True),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(256,activation="relu"),
  layers.Dropout(0.2),
  layers.Dense(128,activation="relu"),
  layers.Dropout(0.2),
  layers.Dense(64,activation="relu"),
  layers.Dropout(0.2),
  layers.Dense(32, activation="relu"),
  layers.Dropout(0.2),
  layers.Dense(nb_classes,activation="softmax")
])

model.summary()

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer='adam',
              metrics=tf.keras.metrics.Accuracy())

epochs = 50
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=epochs)

model.save("Model.h5")
print(model.predict(X_train[:1]))


