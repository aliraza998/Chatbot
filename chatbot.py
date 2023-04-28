import nltk
import numpy
import tensorflow
import tflearn
import random
from nltk.stem.lancaster import LancasterStemmer

nltk.download('punkt')

import json

with open('intents.json') as file:
    data = json.load(file)  # loads the contents of the file into a dictionary data.

stemmer = LancasterStemmer()  # creates an instance of the LancasterStemmer class.

words = []  # is a list to hold all the words from the patterns in the intents.json file.
labels = []  # is a list to hold the unique tags of the intents.
docs_x = []  # is a list to hold the tokenized words of the patterns.
docs_y = []  # is a list to hold the corresponding tags of the patterns.

for intent in data['intents']:  # loops through each intent in the 'data' dictionary.
    for pattern in intent['patterns']:  # loops through each pattern of the current intent.
        wrds = nltk.word_tokenize(
            pattern)  # tokenizes the words in the pattern using the word_tokenize function from nltk.
        words.extend(wrds)  # adds the tokenized words to the words list.
        docs_x.append(wrds)  # adds the tokenized words to the docs_x list.
        docs_y.append(intent["tag"])  # adds the tag of the current intent to the docs_y list.

    if intent[
        'tag'] not in labels:  # checks if the tag of the current intent is not in the labels list. If so, it adds the tag to the labels list.
        labels.append(intent['tag'])

# Stem and lowercase the words in the words list and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

# Sort the labels list
labels = sorted(labels)

# Initialize empty lists for holding the training data and output data
training = []
output = []

# Create an empty output row for each label
out_empty = [0 for _ in range(len(labels))]

# Loop through each tokenized words in docs_x
for x, doc in enumerate(docs_x):
    bag = []
    # Stem and lowercase the words in the tokenized words
    wrds = [stemmer.stem(w.lower()) for w in doc]

    # Create the bag of words representation of the tokenized words
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    # Create the output row for the current tokenized words by updating the value in the out_empty list
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    # Add the bag of words representation and the output row to the training and output lists
    training.append(bag)
    output.append(output_row)

# Convert the training and output lists to numpy arrays
training = numpy.array(training)
output = numpy.array(output)

# Reset the tensorflow graph
tensorflow.compat.v1.reset_default_graph()

# Define the neural network structure
# input layer with shape [None, length of training data]
net = tflearn.input_data(shape=[None, len(training[0])])
# fully connected layer with 8 neurons
net = tflearn.fully_connected(net, 8)
# another fully connected layer with 8 neurons
net = tflearn.fully_connected(net, 8)
# final fully connected layer with output size equal to length of output data and activation function "softmax"
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
# regression layer
net = tflearn.regression(net)

# Create a Deep Neural Network (DNN) model using the defined structure
model = tflearn.DNN(net)
# Fit the model to the training data with 1000 epochs, batch size of 8 and display metrics during training
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
# Save the trained model
model.save("model.tflearn")


# Define a function to convert a given input string into a bag of words representation
def bag_of_words(s, words):
    # initialize a bag of zeros with length equal to the number of words
    bag = [0 for _ in range(len(words))]

    # tokenize the input string into words
    s_words = nltk.word_tokenize(s)
    # stem and lowercase each word
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    # for each word in the input string
    for se in s_words:
        # for each word in the list of words
        for i, w in enumerate(words):
            # if a match is found, set the corresponding element in the bag to 1
            if w == se:
                bag[i] = 1
    # return the bag as a numpy array
    return numpy.array(bag)


# Defined a function to chat with the bot
def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        # Predict the result using the model and the bag of words representation of the input
        results = model.predict([bag_of_words(inp, words)])
        # Get the index of the highest result
        results_index = numpy.argmax(results)
        # Get the corresponding tag
        tag = labels[results_index]

        # Search for the responses corresponding to the tag in the intents
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        # Print a random response from the list of responses
        print(random.choice(responses))


# Calling the chat function to start the conversation
chat()