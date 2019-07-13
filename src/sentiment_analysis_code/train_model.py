"""
Main file with loading data and training RNN for sentiment twitter analysis
"""

from __future__ import print_function

import os
import sys
import pickle
from keras.layers import Embedding
from keras.models import load_model
from keras.initializers import Constant
from models import get_bidirectional_LSTM_model
from nlp_utilities import path_builder, index_word_vectors, get_data, split_data, prepare_embedding_matrix, \
    macro_averaged_recall_tf_onehot, \
    macro_averaged_recall_tf_soft, gpu_configuration_initialization, macro_averaged_precision, macro_averaged_f1

from config import config, model_config, DEBUG

# Import callbacks that will be passed to the fit functions
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Import custom objects that were created for this task
from logger_to_file import Logger
from keras.utils.generic_utils import get_custom_objects

# ----------------------------------------------------------------------------------------------------------------------
# region main

if __name__ == "__main__":
    # Set up environmental variables etc.
    continue_training = 0
    # Name of the training session to which all objects will be saved
    NAME_OF_THE_TRAIN_SESSION = config["NAME_OF_THE_TRAIN_SESSION"]
    PATH_TO_THE_LEARNING_SESSION = "./learning_sessions/" + NAME_OF_THE_TRAIN_SESSION + "/"

    # Name of the pretrained model, if continue_training=1
    pretrained_filepath = PATH_TO_THE_LEARNING_SESSION + config["MODEL_NAME"]

    print("DEBUG = {} \nconfig = {}\nmodel_config = {}".format(DEBUG, config, model_config))

    # Update custom object with my own loss functions
    custom_objects = {'macro_averaged_recall_tf_onehot': macro_averaged_recall_tf_onehot,
                      'macro_averaged_recall_tf_soft': macro_averaged_recall_tf_soft
                      }
    get_custom_objects().update(custom_objects)

    # Configuring gpu for the training
    gpu_configuration_initialization()

    # Build directories
    path_builder(PATH_TO_THE_LEARNING_SESSION)
    # Create logger to files (copying std out stream to a file)
    sys.stdout = Logger(PATH_TO_THE_LEARNING_SESSION + "log_training")

    # First, build index mapping words in the embeddings set to their embedding vector
    print("Indexing word vectors.")

    # Using twitter specific dataset. Pretrained by glove.
    filename_to_read = config["TWITTER_GLOVE"]
    embeddings_index = index_word_vectors(filename_to_read, **config)
    print("Found {} word vectors.".format(len(embeddings_index)))

    # Second, prepare text samples and their labels
    print("Processing text dataset")

    train_directory = config["TRAIN_DIRECTORY"]
    test_directory = config["TEST_DIRECTORY"]

    # Third, vectorize the training text samples into a 2D integer tensor
    if config["DATA_PREPROCESSED"]:
        with open('data_combiner_train.pickle', 'rb') as handle:
            data_combiner = pickle.load(handle)
            data = data_combiner[0]
            labels = data_combiner[1]
            word_index = data_combiner[2]
            tokenizer = data_combiner[3]
            texts = data_combiner[4]
        with open('data_combiner_test.pickle', 'rb') as handle_test:
            data_combiner = pickle.load(handle_test)
            test_data = data_combiner[0]
            test_labels = data_combiner[1]
            test_word_index = data_combiner[2]
            test_tokenizer = data_combiner[3]
            test_texts = data_combiner[4]
    else:
        data, labels, word_index, tokenizer, texts = get_data(train_directory, config, tokenizer=None, mode="training")

        # Fourth, vectorize the testing set using provided tokenizer
        test_data, test_labels, test_word_index, test_tokenizer, test_texts = get_data(test_directory, config,
                                                                                       tokenizer=tokenizer,
                                                                                       mode="test")
    # split the data into a training set and a validation set
    x_train, y_train, x_val, y_val = split_data(data, labels, **config)


    # Fifth, prepare embedding matrix
    embedding_matrix, num_words = prepare_embedding_matrix(word_index, embeddings_index, **config)

    # Sixth, load pre-trained word embeddings into an Embedding layer. Additionally- keep the embeddings fixed.
    embedding_layer = Embedding(num_words,
                                config["EMBEDDING_DIM"],
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=config["MAX_SEQUENCE_LENGTH"],
                                trainable=False)

    # Seventh, build the model with the use of the Embedding layer
    model = get_bidirectional_LSTM_model(embedding_layer, macro_averaged_recall_tf_onehot, macro_averaged_recall_tf_soft, macro_averaged_precision, macro_averaged_f1)

    # If there is a desire to continue training of a model, this may be used
    if continue_training:
        model = load_model(pretrained_filepath)

        print("Successfully loaded pretrained model {}".format(pretrained_filepath))

    # Eighth, set callbacks
    callbacks = []

    # Create model checkpointer to save best models during training
    model_checkpoint = ModelCheckpoint(PATH_TO_THE_LEARNING_SESSION + 'model_ckpt.h5',
                                       monitor='val_loss',
                                       save_weights_only=False,
                                       save_best_only=True)
    callbacks.append(model_checkpoint)

    # Callback for stopping the training if no progress is achieved
    early_stopper = EarlyStopping(monitor='val_loss', patience=2)
    callbacks.append(early_stopper)

    print("Fitting normal way ...")
    history = model.fit(data, labels,
                        batch_size=config["BATCH_SIZE"],
                        epochs=config["EPOCHS"],
                        validation_data=(test_data, test_labels),
                        callbacks=callbacks)

    print("Fitting normal way Finished...!")
    model.save(PATH_TO_THE_LEARNING_SESSION + config["MODEL_NAME"])

    loss_metrics_eval = model.evaluate(x=test_data, y=test_labels, batch_size=config["BATCH_SIZE"])
    print("Evaluated metrics = {} \n {}".format(loss_metrics_eval, model.metrics_names))

    print("Script finished successfully!")
# endregion main
# ----------------------------------------------------------------------------------------------------------------------
