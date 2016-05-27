import logging
import time
import numpy as np
import theano.tensor as T
import theano
import lasagne


logger = logging.getLogger(__name__)


def gen_minibatches(segment_container_gen,
        classes,
        batch_size,
        num_features,
        num_time_bins,
        feature_name):
    """Generates batches of segments from segment container generator"""

    if num_features == 1:
        batch = np.empty((batch_size, 1, num_time_bins), dtype=np.float32)
    else:
        batch = np.empty((batch_size, 1, num_features, num_time_bins), dtype=np.float32)
    targets = np.empty((batch_size), dtype=np.int16)

    count = 0
    for sc in segment_container_gen.execute():
        logger.debug("iterate_minibatch: {}".format(sc.audio_path))
        for s in sc.segments:
            if not feature_name in s.features:
                break
            if s.activity:
                if num_features == 1:
                    batch[count, 0, :] = s.features[feature_name].T # sort out shape
                else:
                    batch[count, 0, :, :] = s.features[feature_name].T # sort out shape
                targets[count] = classes.index(s.label)
                count += 1
                if count == batch_size:
                    count = 0
                    yield batch, targets


# raw audio
def build_cnn_audio(batch_size, num_time_bins, num_classes):

    # create theano vars
    input_var = T.tensor3('inputs') # batch_size x num_channels x num_time_bins
    target_var = T.ivector('targets')

    # input layer
    network = lasagne.layers.InputLayer(
            shape=(batch_size, 1, num_time_bins),
            input_var=input_var)
    
    # conv layer
    cl1_num_filters = 20
    cl1_filter_size = 400
    network = lasagne.layers.Conv1DLayer(network, cl1_num_filters, cl1_filter_size,
            stride=1,
            pad=0,
            untie_biases=False,
            W=lasagne.init.GlorotUniform(gain="relu"), # # TODO: make sure it's scaled according to input_size
            b=lasagne.init.Constant(0.),
            nonlinearity=lasagne.nonlinearities.rectify, # TODO apply that after pooling to save some useless calculation
            flip_filters=True, # we do want convolution
            convolution=lasagne.theano_extensions.conv.conv1d_mc0) # this uses nnet.conv2d, supposed to take advantages of cuDNN

    # pooling layer
    pl1_pool_size = 4
    network = lasagne.layers.Pool1DLayer(network,
            pl1_pool_size,
            stride=None,
            pad=0,
            ignore_border=True,
            mode='max') #TODO check mean vs max

    # conv layer
    cl2_num_filters = 20
    cl2_filter_size = 100
    network = lasagne.layers.Conv1DLayer(network, cl2_num_filters, cl2_filter_size,
            stride=1,
            pad=0,
            untie_biases=False,
            W=lasagne.init.GlorotUniform(gain="relu"), # # TODO: make sure it's scaled according to input_size
            b=lasagne.init.Constant(0.),
            nonlinearity=lasagne.nonlinearities.rectify, # TODO apply that after pooling to save some useless calculation
            flip_filters=True, # we do want convolution
            convolution=lasagne.theano_extensions.conv.conv1d_mc0) # this uses nnet.conv2d, supposed to take advantages of cuDNN

    # pooling layer
    pl2_pool_size = 4
    network = lasagne.layers.Pool1DLayer(network, pl2_pool_size,
            stride=None,
            pad=0,
            ignore_border=True,
            mode='max') #TODO check mean vs max

    # dense layer
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.1),
            num_units=400,
            W=lasagne.init.GlorotUniform(gain="relu"), # # TODO: make sure it's scaled according to input_size
            b=lasagne.init.Constant(0.),
            nonlinearity=lasagne.nonlinearities.rectify)

    # softmax
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.1),
        num_units=num_classes,
        nonlinearity=lasagne.nonlinearities.softmax)

    return input_var, target_var, network


# mel bands
def build_cnn_mel(batch_size, num_features, num_time_bins, num_classes):

    # create theano vars
    input_var = T.tensor4('inputs') # batch_size x num_channels x num_time_bins
    target_var = T.ivector('targets')

    # input layer
    network = lasagne.layers.InputLayer(
            shape=(batch_size, 1, num_features, num_time_bins),
            input_var=input_var)
    
    # conv layer
    cl1_num_filters = 20
    cl1_filter_size = (8,10)
    network = lasagne.layers.Conv2DLayer(network, cl1_num_filters, cl1_filter_size,
            stride=1,
            pad=0,
            untie_biases=False,
            W=lasagne.init.GlorotUniform(gain="relu"), # # TODO: make sure it's scaled according to input_size
            b=lasagne.init.Constant(0.),
            nonlinearity=lasagne.nonlinearities.rectify, # TODO apply that after pooling to save some useless calculation
            convolution=theano.tensor.nnet.conv2d)

    # pooling layer
    pl1_pool_size = 2
    network = lasagne.layers.Pool2DLayer(network,
            pl1_pool_size,
            stride=None,
            pad=0,
            ignore_border=True,
            mode='max') #TODO check mean vs max

    # conv layer
    cl2_num_filters = 20
    cl2_filter_size = (8,10)
    network = lasagne.layers.Conv2DLayer(network, cl2_num_filters, cl2_filter_size,
            stride=1,
            pad=0,
            untie_biases=False,
            W=lasagne.init.GlorotUniform(gain="relu"), # # TODO: make sure it's scaled according to input_size
            b=lasagne.init.Constant(0.),
            nonlinearity=lasagne.nonlinearities.rectify, # TODO apply that after pooling to save some useless calculation
            flip_filters=True, # we do want convolution
            convolution=theano.tensor.nnet.conv2d)

    # pooling layer
    pl2_pool_size = 2
    network = lasagne.layers.Pool2DLayer(network, pl2_pool_size,
            stride=None,
            pad=0,
            ignore_border=True,
            mode='max') #TODO check mean vs max

    # dense layer
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.1),
            num_units=200,
            W=lasagne.init.GlorotUniform(gain="relu"), # # TODO: make sure it's scaled according to input_size
            b=lasagne.init.Constant(0.),
            nonlinearity=lasagne.nonlinearities.rectify)

    # softmax
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.1),
        num_units=num_classes,
        nonlinearity=lasagne.nonlinearities.softmax)

    return input_var, target_var, network


# chirplets
def build_cnn_chirplets(batch_size, num_features, num_time_bins, num_classes):

    # create theano vars
    input_var = T.tensor4('inputs') # batch_size x num_channels x num_time_bins
    target_var = T.ivector('targets')

    # input layer
    network = lasagne.layers.InputLayer(
            shape=(batch_size, 1, num_features, num_time_bins),
            input_var=input_var)
    
    # conv layer
    cl1_num_filters = 20
    cl1_filter_size = (8,10)
    network = lasagne.layers.Conv2DLayer(network, cl1_num_filters, cl1_filter_size,
            stride=1,
            pad=0,
            untie_biases=False,
            W=lasagne.init.GlorotUniform(gain="relu"), # # TODO: make sure it's scaled according to input_size
            b=lasagne.init.Constant(0.),
            nonlinearity=lasagne.nonlinearities.rectify, # TODO apply that after pooling to save some useless calculation
            convolution=theano.tensor.nnet.conv2d)

    # pooling layer
    pl1_pool_size = 2
    network = lasagne.layers.Pool2DLayer(network,
            pl1_pool_size,
            stride=None,
            pad=0,
            ignore_border=True,
            mode='max') #TODO check mean vs max

    # conv layer
    cl2_num_filters = 20
    cl2_filter_size = (8,10)
    network = lasagne.layers.Conv2DLayer(network, cl2_num_filters, cl2_filter_size,
            stride=1,
            pad=0,
            untie_biases=False,
            W=lasagne.init.GlorotUniform(gain="relu"), # # TODO: make sure it's scaled according to input_size
            b=lasagne.init.Constant(0.),
            nonlinearity=lasagne.nonlinearities.rectify, # TODO apply that after pooling to save some useless calculation
            flip_filters=True, # we do want convolution
            convolution=theano.tensor.nnet.conv2d)

    # pooling layer
    pl2_pool_size = 2
    network = lasagne.layers.Pool2DLayer(network, pl2_pool_size,
            stride=None,
            pad=0,
            ignore_border=True,
            mode='max') #TODO check mean vs max

    # dense layer
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.1),
            num_units=200,
            W=lasagne.init.GlorotUniform(gain="relu"), # # TODO: make sure it's scaled according to input_size
            b=lasagne.init.Constant(0.),
            nonlinearity=lasagne.nonlinearities.rectify)

    # softmax
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.1),
        num_units=num_classes,
        nonlinearity=lasagne.nonlinearities.softmax)

    return input_var, target_var, network


# chirplets - no pooling
def build_cnn_chirplets_no_pooling(batch_size, num_features, num_time_bins, num_classes):

    # create theano vars
    input_var = T.tensor4('inputs') # batch_size x num_channels x num_time_bins
    target_var = T.ivector('targets')

    # input layer
    network = lasagne.layers.InputLayer(
            shape=(batch_size, 1, num_features, num_time_bins),
            input_var=input_var)
    
    # conv layer
    cl1_num_filters = 20
    cl1_filter_size = (8,10)
    network = lasagne.layers.Conv2DLayer(network, cl1_num_filters, cl1_filter_size,
            stride=1,
            pad=0,
            untie_biases=False,
            W=lasagne.init.GlorotUniform(gain="relu"), # # TODO: make sure it's scaled according to input_size
            b=lasagne.init.Constant(0.),
            nonlinearity=lasagne.nonlinearities.rectify, # TODO apply that after pooling to save some useless calculation
            convolution=theano.tensor.nnet.conv2d)

    # conv layer
    cl2_num_filters = 20
    cl2_filter_size = (8,10)
    network = lasagne.layers.Conv2DLayer(network, cl2_num_filters, cl2_filter_size,
            stride=1,
            pad=0,
            untie_biases=False,
            W=lasagne.init.GlorotUniform(gain="relu"), # # TODO: make sure it's scaled according to input_size
            b=lasagne.init.Constant(0.),
            nonlinearity=lasagne.nonlinearities.rectify, # TODO apply that after pooling to save some useless calculation
            flip_filters=True, # we do want convolution
            convolution=theano.tensor.nnet.conv2d)

    # dense layer
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.1),
            num_units=500,
            W=lasagne.init.GlorotUniform(gain="relu"), # # TODO: make sure it's scaled according to input_size
            b=lasagne.init.Constant(0.),
            nonlinearity=lasagne.nonlinearities.rectify)

    # softmax
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.1),
        num_units=num_classes,
        nonlinearity=lasagne.nonlinearities.softmax)

    return input_var, target_var, network


def train_cnn(sc_gen_train,
        sc_gen_valid,
        sc_gen_test,
        classes,
        feature_name,
        network,
        input_var,
        target_var,
        num_epochs,
        batch_size,
        num_features,
        num_time_bins,
        learning_rate,
        reg_coef):
    
    # define objective
    train_prediction = lasagne.layers.get_output(network)
    train_loss = lasagne.objectives.categorical_crossentropy(train_prediction, target_var)
    train_loss = train_loss.mean() + reg_coef * lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)

    # define updates
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            train_loss, params, learning_rate=learning_rate, momentum=0.9)

    # define validation set evaluation
    valid_prediction = lasagne.layers.get_output(network, deterministic=True) # deterministic=True disables the dropout layers
    valid_loss = lasagne.objectives.categorical_crossentropy(valid_prediction,
            target_var)
    valid_loss = valid_loss.mean()
    valid_acc = T.mean(T.eq(T.argmax(valid_prediction, axis=1), target_var),
            dtype=theano.config.floatX)

    # define test set evaluation
    test_prediction = lasagne.layers.get_output(network, deterministic=True) # deterministic=True disables the dropout layers
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
            dtype=theano.config.floatX)

    logger.debug("Compiling theano's expressions...")

    # compile Theano's expressions
    train_fn = theano.function([input_var, target_var], train_loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [valid_loss, valid_acc])
    test_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    logger.debug("Done")

    # training loop
    for epoch in range(num_epochs):

        # train
        sc_gen_train.reset()
        train_err = 0
        train_batches = 0
        start_time = time.time()
        train_batch_gen = gen_minibatches(sc_gen_train,
                classes,
                batch_size,
                num_features,
                num_time_bins,
                feature_name)
        for inputs, targets in train_batch_gen:
            train_err += train_fn(inputs, targets)
            train_batches += 1
        
        # And a full pass over the validation data:
        sc_gen_valid.reset()
        val_err = 0
        val_acc = 0
        val_batches = 0
        valid_batch_gen = gen_minibatches(sc_gen_valid,
                classes,
                batch_size,
                num_features,
                num_time_bins,
                feature_name)
        for inputs, targets in valid_batch_gen:
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
        
        # And a full pass over the test data:
        sc_gen_test.reset()
        test_err = 0
        test_acc = 0
        test_batches = 0
        test_batch_gen = gen_minibatches(sc_gen_test,
                classes,
                batch_size,
                num_features,
                num_time_bins,
                feature_name)
        for inputs, targets in test_batch_gen:
            err, acc = test_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1


        # Then we print the results for this epoch:
        logger.info("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        logger.info("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        logger.info("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        logger.info("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        logger.info("  test loss:\t\t{:.6f}".format(test_err / test_batches))
        logger.info("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))
