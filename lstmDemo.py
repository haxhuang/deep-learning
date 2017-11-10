from keras.optimizers import Adam
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras import callbacks, utils

docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.',
        'fuck you',
        'suck',
        'Good boy',
        'Bad body',
        'fuck']  # define class labels
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
pred_inputs = ["well done", "fuck you", "fuck you mum", "not good", "Great effort"]
vocab_size = len(docs) + 1
max_length = 6


def padd_docs(texts):
    # integer encode the documents
    encoded_docs = [one_hot(d, vocab_size) for d in texts]
    # print(encoded_docs)
    # pad documents to a max length of 4 words
    padded_seqs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    return padded_seqs


def padd_docs_v1(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    seq = tokenizer.texts_to_sequences(texts)
    print("Words:",len(tokenizer.word_counts))
    padded_seqs = pad_sequences(seq, maxlen=max_length, padding='post')
    return padded_seqs


train_docs = padd_docs_v1(docs)
# print("one-hot:", train_docs)
train_docs2 = padd_docs_v1(docs)
# print("tokenizer:", train_docs2)
predict_docs = padd_docs_v1(pred_inputs)
model = Sequential()

X_train = train_docs[:13]
Y_train = labels[:13]
X_test = train_docs[13:]
Y_test = labels[13:]


def train_lstm():
    model.add(Embedding(21, 256))
    model.add(LSTM(128, activation='relu'))  # try using a GRU instead, for fun
    # model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])
    # print(model.summary())
    tbCallBack = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    model.fit(train_docs, labels, batch_size=4, epochs=100, verbose=1, validation_split=0.2, callbacks=[tbCallBack])
    loss, accuracy = model.evaluate(train_docs, labels, verbose=1)
    print('Accuracy: %f' % (accuracy * 100))


def train():
    # model.add(Embedding(vocab_size, 8, input_length=max_length))
    model.add(Embedding(32, 64, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())
    model.fit(X_train, Y_train, epochs=100, verbose=0, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


print("Train shape:", train_docs.shape)
train_lstm()
# train()

pred = model.predict(predict_docs)
classes = model.predict_classes(predict_docs)
# acc = np_utils.accuracy(classes, labels)
print("Predict result:", classes)
