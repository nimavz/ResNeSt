# develop a classifier for the 5 Celebrity Faces Dataset
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
# load dataset
data = load('dataset_embeddings.npz')
trainX, trainy = data['arr_0'], data['arr_1']
# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
predict
yhat_train = model.predict(trainX)
# score
score_train = accuracy_score(trainy, yhat_train)
# summarize
print('Accuracy: train=%.3f' % (score_train*100))