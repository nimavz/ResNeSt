def get_model(model_name='ResNest50', input_shape=(224, 224, 3), n_classes=256,
              verbose=False, dropout_rate=0, fc_activation=None, **kwargs):
    '''get_model
    input_shape: (h,w,c)
    fc_activation: sigmoid,softmax
    '''
    model_name = model_name.lower()

    resnest_parameters = {
        'resnest50': {
            'blocks_set': [3, 4, 6, 3],
            'stem_width': 224,
        },
        'resnest101': {
            'blocks_set': [3, 4, 23, 3],
            'stem_width': 64,
        },
        'resnest200': {
            'blocks_set': [3, 24, 36, 3],
            'stem_width': 64,
        },
        'resnest269': {
            'blocks_set': [3, 30, 48, 8],
            'stem_width': 64,
        },
    }

    if model_name in resnest_parameters.keys():
        model = ResNest(verbose=verbose, input_shape=input_shape,
                        n_classes=n_classes, dropout_rate=dropout_rate, fc_activation=fc_activation,
                        blocks_set=resnest_parameters[model_name]['blocks_set'], radix=2, groups=1, bottleneck_width=64,
                        deep_stem=True,
                        stem_width=resnest_parameters[model_name]['stem_width'], avg_down=True, avd=True,
                        avd_first=False, **kwargs).build()

    else:
        raise ValueError('Unrecognize model name {}'.format(model_name))
    return model


if __name__ == "__main__":

    # model_names = ['ResNest50','ResNest101','ResNest200','ResNest269']
    model_names = ['ResNest50']
    # model_names = ['RegNetX400','RegNetX1.6','RegNetY400','RegNetY1.6']
    input_shape = [224, 224, 3]
    n_classes = 256
    fc_activation = 'softmax'  # softmax sigmoid

    for model_name in model_names:
        print('model_name', model_name)
        model = get_model(model_name=model_name, input_shape=input_shape, n_classes=n_classes,
                          verbose=True, fc_activation=fc_activation)
        print('-' * 10)


model_name = 'ResNest50'
input_shape = [224,224,3]
n_classes = 256
fc_activation = 'softmax'
active = 'relu' # relu or mish

model = get_model(model_name=model_name,
                  input_shape=input_shape,
                  n_classes=n_classes,
                  fc_activation=fc_activation,
                  active=active,
                  verbose=False,
                 )

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy())


model.fit(train_X, train_y, epochs=10, batch_size=30)


from sklearn.metrics import accuracy_score
yhat_train = model.predict(train_X)
# score
score_train = accuracy_score(train_y, np.round(yhat_train), normalize=False)
# summarize
print('Accuracy: train=%.3f' % (score_train*100))

# calculate a face embedding for each face in the dataset using facenet
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from tensorflow.keras.models import load_model


# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]


# load the face dataset
data = load('dataset_n.npz')
trainX, trainy = data['arr_0'], data['arr_1']
trainX = [i for i in trainX if i is not None]
trainy = trainy[0:2034]
# load the facenet model
print('Loaded Model')
# convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX:
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)
newTrainX = asarray(newTrainX)
print(newTrainX.shape)
# save arrays to one file in compressed format
savez_compressed('dataset_embeddings.npz', newTrainX, trainy)