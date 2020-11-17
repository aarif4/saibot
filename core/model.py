"""
    In this module, we'll create a CNN with Tensorflow using keras
"""
import tensorflow as tf
import random
import os
import numpy as np
from sklearn.model_selection import train_test_split

def setup_model(learning_rate):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3,3), padding='same',
                                    input_shape=(176,200,3),
                                    activation='relu'))
    model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(64, (3,3), padding='same',
                                    activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(128, (3,3), padding='same',
                                    activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(4, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=learning_rate, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    return model

def run_model(model,learning_rate):
    #logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    tb = tf.keras.callbacks.TensorBoard(log_dir='logs/stage1')

    train_data_dir = "train_data" # TODO: Fix this to be better

    # want to load in a few games at a time
    hm_epochs = 10
    for i in range(hm_epochs):
        current = 0
        increment = 20 # was 200
        not_maximum = True
        all_files = os.listdir(train_data_dir)
        maximum = len(all_files)
        random.shuffle(all_files)
        while not_maximum:
            print("Currently doing {}:{}".format(current,current+increment))
            no_attacks = []
            attack_closest_to_nexus = []
            attack_enemy_structures = []
            attack_enemy_start = []
            for file in all_files[current:current+increment]:
                full_path = os.path.join(train_data_dir, file)
                data = list(np.load(full_path, allow_pickle=True))
                for d in data:
                    choice = np.argmax(d[0])
                    if choice == 0:
                        no_attacks.append(d)
                    if choice == 1:
                        attack_closest_to_nexus.append(d)
                    if choice == 2:
                        attack_enemy_structures.append(d)
                    if choice == 3:
                        attack_enemy_start.append(d)
                lengths = check_data(no_attacks, attack_closest_to_nexus, attack_enemy_structures, attack_enemy_start)
                lowest_data = min(lengths)

                random.shuffle(no_attacks)
                random.shuffle(attack_closest_to_nexus)
                random.shuffle(attack_enemy_structures)
                random.shuffle(attack_enemy_start)

                # if most of your data is leaning on one element, the nn could learn only that action. So, it's good to balance the input before feeding it in
                no_attacks = no_attacks[:lowest_data]
                attack_closest_to_nexus = attack_closest_to_nexus[:lowest_data]
                attack_enemy_structures = attack_enemy_structures[:lowest_data]
                attack_enemy_start = attack_enemy_start[:lowest_data]

                train_data = no_attacks + attack_closest_to_nexus + attack_enemy_structures + attack_enemy_start
                random.shuffle(train_data)
                test_size = 0.3 # was 100
                batch_size = 50 # was 128

                # TODO: Use tensorflow routines to do train/test split and fit model
                if False:
                    x_train = np.array([i[1] for i in train_data[:-test_size]]).reshape(-1,176, 200, 3) # this is the image/map we make with opencv that has our units and enemy units
                    y_train = np.array([i[0] for i in train_data[:-test_size]])

                    x_test = np.array([i[1] for i in train_data[-test_size:]]).reshape(-1,176, 200, 3)
                    y_test = np.array([i[0] for i in train_data[-test_size:]])
                else:
                    train_data = np.array(train_data)
                    x_train, x_test, y_train, y_test = train_test_split(train_data[:,1], train_data[:,0], test_size=test_size)
                    x_train = np.array(list(x_train))
                    x_test = np.array(list(x_test))
                    y_train = np.array(list(y_train))
                    y_test = np.array(list(y_test))
                    
                
                model.fit(x_train,y_train,
                        batch_size=batch_size,
                        validation_data=(x_test,y_test),
                        shuffle=True, verbose=1, callbacks=[tb])
            
                model.save("BasicCNN-{}-epochs-{}-LR-STAGE1".format(hm_epochs, learning_rate))
                current = current + increment
                if current > maximum:
                    not_maximum = False


def check_data(no_attacks, attack_closest_to_nexus, attack_enemy_structures, attack_enemy_start):
    choices = { "no_attacks":no_attacks,
                "attack_closest_to_nexus":attack_closest_to_nexus,
                "attack_enemy_structures":attack_enemy_structures,
                "attack_enemy_start":attack_enemy_start,
            }
    total_data = 0

    lengths = []
    for choice in choices:
        print("Length of {} is: {}".format(choice, len(choices[choice])))
        total_data = total_data + len(choices[choice])
        lengths.append(len(choices[choice]))
    
    print('Total data length now is:', total_data)
    return lengths


if __name__ == "__main__":
    learning_rate= 0.0001
    model = setup_model(learning_rate)
    run_model(model,learning_rate)