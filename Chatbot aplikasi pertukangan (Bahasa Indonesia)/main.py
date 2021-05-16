import json
import random
import pickle
import tensorflow
import pandas as pd
import tflearn
import numpy
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.python.framework import ops
ops.reset_default_graph()
factory = StemmerFactory()
stemmer = factory.create_stemmer()

nltk.download('punkt')

with open('intents2.json') as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

try:
    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    model.load("model.tflearn")
except:

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)


order = []
perbaikan = ['WC', 'Atap', 'Tembok', 'Pintu', 'Jendela', 'Lantai']
pengecatan = ['Dinding', 'Seng']


def listToString(order):

    str1 = ""

    for ele in order:
        str1 += " "+ele

    return str1


def chat():
    print("Ada yang bisa saya bantu?")
    while True:
        inp = input("Anda: ")

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if tag == "keluar":

            if len(order) == 0:
                print("Semoga hari Anda menyenangkan!")
                break
            else:
                print(
                    "Berikut daftar nama tukang yang kami rekomendasikan untuk layanan yang Anda butuhkan")
                print(filter(order, df))
                print("Sampai jumpa! Semoga hari Anda menyenangkan!")
                break

        if tag == 'cat dinding' and 'memulas dinding' not in order:
            order.append('Dinding')
        if tag == 'cat atap' and 'memulas atap' not in order:
            order.append('Seng')

        if tag == 'atap' and 'bumbung' not in order:
            order.append('Atap')

        if tag == 'tembok' and 'dinding' not in order:
            order.append('Tembok')

        if tag == 'pintu' and 'pintu' not in order:
            order.append('Pintu')

        if tag == 'jendela' and 'jendela' not in order:
            order.append('Jendela')

        if tag == 'wc' and 'wc' not in order:
            order.append('WC')

        if tag == 'lantai' and 'ubin' not in order:
            order.append('Lantai')

        for tg in data["intents"]:

            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))


def reccomend(inp):
    results = model.predict([bag_of_words(inp, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']

    print(random.choice(responses))


def filter(order, df):

    if order[0] in perbaikan:
        for i in order:
            df_filtered = df[df['layananPerbaikan'].str.contains(
                i, regex=False, case=False, na=False)]
            df_filtered = df[df['layananPengecatan'].str.contains(
                i, regex=False, case=False, na=False)]
    else:
        for i in order:
            df_filtered = df[df['layananPengecatan'].str.contains(
                i, regex=False, case=False, na=False)]
            df_filtered = df[df['layananPerbaikan'].str.contains(
                i, regex=False, case=False, na=False)]
    return df_filtered.where(df['layananRating'] > 4.0).dropna()


df = pd.read_csv('capsdata.csv')

chat()
