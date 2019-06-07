import json
import codecs
import tensorflow as tf
import numpy as np
import os, string, re
from two_way_dict import TwoWayDict

# different feature attempts
# use first 3 and last 3 words of the 1st sentence and first 3 words of the 2nd sentence as features
l3f3 = True
# use (x,y) for x in Arg1 and y in Arg2 as features
word_pairs = False

# generate output output_parser_file_name.json based on current neural network for evaluation
parse_output = True
output_parser_file_name = "output4"


def output_parser(output_matrix):
    pdtb_file = codecs.open('test/relations.json', encoding='utf8')
    if type_mode == 2:
        relations = [json.loads(x) for x in pdtb_file if json.loads(x)["Type"] == "Explicit"]
    elif type_mode == 1:
        relations = [json.loads(x) for x in pdtb_file if json.loads(x)["Type"] == "Implicit"]
    else:
        relations = [json.loads(x) for x in pdtb_file]

    for i, relation in enumerate(relations):
        sense = []
        for j, pivot in enumerate(output_matrix[i]):
            if pivot == 1:
                sense.append(label_2wdict[j])
        if not sense:  # no sense extracted (rare problem)
            sense.append("Expansion.Conjunction")  # apply an arbitrary value
        data = {
            'DocID': relation["DocID"],
            'Arg1': {
                'TokenList': [x for [_, _, x, _, _] in relation["Arg1"]["TokenList"]]
            },
            'Arg2': {
                'TokenList': [x for [_, _, x, _, _] in relation["Arg2"]["TokenList"]]
            },
            'Connective': {
                'TokenList': [x for [_, _, x, _, _] in relation["Connective"]["TokenList"]]
            },
            'Sense': sense,
            'Type': relation["Type"]
        }

        data_file = json.dumps(data)
        # data_indented = json.dumps(d, indent=4)

        with open("test/" + output_parser_file_name + ".json", "a") as f:
            f.write(data_file)
            f.write('\n')


def strip_word(word):
    pattern = r"[{}]".format(string.punctuation)
    new_word = re.sub(pattern, '', word).lower()
    return new_word


def init_parameters():
    word_list = set()
    labels = []

    for fn in os.listdir('train/raw'):
        file = open("train/raw/" + fn, "r", encoding="ISO-8859-1")
        for line in file:
            for word in line.split():
                word_list.add(strip_word(word))

    pdtb_file = codecs.open('train/relations.json', encoding='utf8')
    relations = [json.loads(x) for x in pdtb_file]

    filtered_word_list = set()
    print("Generating word_list ...")
    for count, relation in enumerate(relations):
        if count % (int(len(relations) / 5)) == 0:
            print("Progress:", "{:.0%}".format(count / len(relations)))
        if word_pairs:
            word_list.add((strip_word(arg1), strip_word(arg2)) for arg1 in relation["Arg1"]["RawText"] for arg2 in
                          relation["Arg2"]["RawText"])
        for i in relation["Sense"]:
            if i not in labels:
                labels.append(i)
        for (_, _, j, _, _) in relation["Connective"]["TokenList"]:
            filtered_word_list.add(j)

    return word_list, labels


word_list, labels = init_parameters()

# hyper-parameters
learning_rate = 0.1
batch_size = 2000  # mini-batch size for training
n_hidden_1 = 256  # number of neurons for the 1st layer
n_hidden_2 = 256  # number of neurons for the 2nd layer
num_inputs = len(word_list)  # total words
num_classes = len(labels)  # total senses
display_step = 1  # print each x step

# implicit = 1, explicit = 2 and all relations = 3
type_mode = 3

# tf input
X = tf.placeholder("float", [None, num_inputs])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight and bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_inputs, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create a neural network model with 3 layers
def neural_net(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# apply non-linear function
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# get loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# get accuracy
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize the variables
init = tf.global_variables_initializer()


# parse the data sets in json into matrices
def parse_to_matrix(dir):
    print("Generating initial matrices for " + dir + " data set ...")
    pdtb_file = codecs.open(dir + '/relations.json', encoding='utf8')
    if type_mode == 2:
        relations = [json.loads(x) for x in pdtb_file if json.loads(x)["Type"] == "Explicit"]
    elif type_mode == 1:
        relations = [json.loads(x) for x in pdtb_file if json.loads(x)["Type"] == "Implicit"]
    else:
        relations = [json.loads(x) for x in pdtb_file]

    input_matrix_x = np.zeros((len(relations), num_inputs))
    input_matrix_y = np.zeros((len(relations), num_classes))

    feature_2wdict = TwoWayDict()
    label_2wdict = TwoWayDict()

    for i, label in enumerate(labels):
        label_2wdict[i] = label
    for i, feature in enumerate(word_list):
        feature_2wdict[i] = feature

    for num_relation, relation in enumerate(relations):
        if num_relation % (int(len(relations) / 5)) == 0:
            print("Progress:", "{:.0%}".format(num_relation / len(relations)))
        total1 = len(relation["Arg1"]["RawText"].split())
        total2 = len(relation["Arg2"]["RawText"].split())
        for i, word in enumerate(relation["Arg1"]["RawText"].split()):
            if (i > total1 - 4) or not l3f3:
                new_word = strip_word(word)
                if new_word in word_list:
                    input_matrix_x[num_relation, feature_2wdict[new_word]] += 1
        for j, word in enumerate(relation["Arg2"]["RawText"].split()):
            if (j < 3) or not l3f3:
                new_word = strip_word(word)
                if new_word in word_list:
                    input_matrix_x[num_relation, feature_2wdict[new_word]] += 1
        if word_pairs:
            for i, word in enumerate(relation["Arg1"]["RawText"].split()):
                for j, word in enumerate(relation["Arg2"]["RawText"].split()):
                    if (i, j) in word_list:
                        input_matrix_x[num_relation, feature_2wdict[(i, j)]] += 1
        for k, word in enumerate(relation["Connective"]["RawText"].split()):
            new_word = strip_word(word)
            if new_word in word_list:
                input_matrix_x[num_relation, feature_2wdict[new_word]] += 1
        for l, c in enumerate(relation["Sense"]):
            input_matrix_y[num_relation, label_2wdict[c]] += 1
    print("... Finished")

    return input_matrix_x, input_matrix_y, label_2wdict


input_matrix_x1, input_matrix_y1, _ = parse_to_matrix('train')
input_matrix_x2, input_matrix_y2, _ = parse_to_matrix('dev')
input_matrix_x3, input_matrix_y3, label_2wdict = parse_to_matrix('test')

# training process
with tf.Session() as sess:
    sess.run(init)

    min_loss = 99999
    last_loss = 0
    converged = False
    converge_counter = 0
    step = 0
    while not converged:
        batch_pivots = np.random.randint(input_matrix_x1.shape[0], size=batch_size)
        batch_x = input_matrix_x1[batch_pivots, :]
        batch_y = input_matrix_y1[batch_pivots, :]

        # back-propagation
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # use dev set to adjust model
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: input_matrix_x2,
                                                                 Y: input_matrix_y2})
            print("Step ", step, ", Loss=", int(loss), ", Accuracy=", "{:.3f}".format(acc))
            # convergent conditions
            if loss < min_loss:
                min_loss = loss
            if loss < last_loss:
                converge_counter = 0
            else:
                converge_counter += 1
            if converge_counter > 1 and min_loss < 500:
                converged = True
        last_loss = loss
        step += 1

    print("... Converged")

    # Calculate accuracy for MNIST test images
    acc, predic = sess.run([accuracy, prediction], feed_dict={X: input_matrix_x3,
                                                              Y: input_matrix_y3})
    print("Test set Accuracy:", acc)
    output_matrix = predic
    if parse_output:
        output_parser(output_matrix)
