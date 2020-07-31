import tensorflow as tf
import numpy as np
import random
import matplotlib
matplotlib.use('TKAgg',warn=False, force=True)
from matplotlib import pyplot as plt

def onehotencode(value):
	minval = 10
	categories = 11
	return [1.0 if i == (value-minval) else 0.0 for i in range(categories)]

x_data = []
y_data = []


datafile = open("../dataProcessing/sidelengths.txt", "r")
rows = datafile.read().split('\n')
for row in rows:
    if(len(row.split(",")) == 5):
        rowdata = [float(elem) for elem in row.split(',')]
        x_data.append(rowdata[:-1])
        y_data.append(rowdata[-1])


n = len(y_data)
split = int(.7*n) # how much to put into train set


#shuffle data
shuffled_indexes = random.sample(range(n), n)

x_temp = []
y_temp = []

for idx in shuffled_indexes:
    x_temp.append(x_data[idx])
    y_temp.append(y_data[idx])

x_data = np.asarray(x_temp)
normalizerx = max([max(row) for row in x_data])
print(normalizerx)
x_data = -1+ (x_data/(.5*normalizerx))

y_data = np.array([val for val in map(onehotencode, y_temp)]) #map 10-20 to onhotencoding
print(y_data, type(y_data), np.shape(y_data))



x_train_data = x_data[:split]
y_train_data = y_data[:split]

test_x_data = x_data[split:]
test_y_data = y_data[split:]


x_train_data = np.reshape(x_train_data, (-1,4))
test_x_data = np.reshape(test_x_data, (-1,4))

input_layer = tf.keras.layers.Input((4,))
hidden_layer_1 = tf.keras.layers.Dense(units=10, activation = tf.keras.activations.relu)(input_layer)
hidden_layer_2 = tf.keras.layers.Dense(units=10, activation = tf.keras.activations.relu)(hidden_layer_1)
hidden_layer_3 = tf.keras.layers.Dense(units=10, activation = tf.keras.activations.relu)(hidden_layer_2)
hidden_layer_4 = tf.keras.layers.Dense(units=10, activation = tf.keras.activations.relu)(hidden_layer_3)
output_layer = tf.keras.layers.Dense(units=11, activation = tf.keras.activations.softmax)(hidden_layer_4)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

print("Enter 'y' to load saved Model.")
if input() == 'y' or input == 'Y':
    model = tf.keras.models.load_model("sidelengthstoDistClassificationMODEL")

wrong =0 
for i in range(1000):
	idx = np.random.randint(0,1000)
	correct = np.argmax(y_data[idx])+10
	predicted = np.argmax(model.predict(np.reshape(x_data[idx], (-1,4))))+10
	if(correct != predicted):
		wrong+=1
		print(wrong, correct)

print("Enter 'y' to continue")
if input() == 'y' or input == 'Y':
    print("continueing")


model.summary()



def vizualize(epochs, logs):
    predicted = model.predict_on_batch(x_data)
    plt.scatter([np.argmax(n)+10 for n in y_data], [np.argmax(n)+10 for n in predicted], marker="+")
    plt.draw()
    plt.pause(0.0001)
    plt.cla()

def scheduler(epoch):
    if(epoch < 400):
        return 1e-2
    else: return 1e-2
    

learnratecallback = tf.keras.callbacks.LearningRateScheduler(scheduler)
vizCallback = tf.keras.callbacks.LambdaCallback(on_epoch_end = vizualize)

model.compile(tf.keras.optimizers.SGD(1e-2), tf.keras.losses.categorical_crossentropy, ["accuracy"])
#model.compile(tf.keras.optimizers.Adam(.1), tf.keras.losses.mean_squared_error, ["accuracy"])
model.fit(x_train_data, y_train_data, batch_size=64, epochs=400, validation_data=[test_x_data, test_y_data], callbacks=[vizCallback, learnratecallback])

print("Enter 'y' to Save Model.")
if input() == 'y' or input == 'Y':
    model.save("sidelengthstoDistClassificationMODEL")
    

