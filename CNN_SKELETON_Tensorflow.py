import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


#capitalized variables are upto you


#DEFINE YOUR MODEL NETWORK IN THIS FUNCTION
def cnn_model_fn(features, labels, mode):

    # Reshape X to 4-D tensor: [batch_size, width, height, channels]

    input_layer = tf.reshape(features["Y"], [-1, WIDTH, HEIGHT, CHANNELS])


    # padding=same/valid
    # Input Tensor Shape: [batch_size, width, height, channel]
    # Output Tensor Shape: [batch_size, width, height, filters]
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=NUM_OF_FILTERS,
      kernel_size=[W, H],       #FILTER_SIZE
      padding="same/valid",       #preserve width and height, if "valid", then width/height will be reduced
      activation=tf.nn.relu)



    # Input Tensor Shape: [batch_size, WIDTH, HEIGHT, FILTERS]
    # Output Tensor Shape: [batch_size, NEW_WIDTH, NEW_HEIGHT, FILTERS]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[W, H], strides=STRIDES)


    # Input Tensor Shape: [batch_size, width, height, channels]
    # Output Tensor Shape: [batch_size, width, height, filters]
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=NUM_OF_FILTERS,
      kernel_size=[W, H],
      padding="same/valid",
      activation=tf.nn.relu)

    # Input Tensor Shape: [batch_size, WIDTH, HEIGHT, FILTERS]
    # Output Tensor Shape: [batch_size, NEW_WIDTH, NEW_HEIGHT, FILTERS]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[W, H], strides=STRIDES)




    #######   CREATE OTHER LAYERS AS NEEDED   #############




    # Flatten tensor into a batch of vectors
    # Merge all features and create a large vector
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    #Fully connected layer
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
      inputs=dense, rate=DROP_RATE, training= (mode == tf.estimator.ModeKeys.TRAIN))

    # FINAL OUTPUT LAYER
    logits = tf.layers.dense(inputs=dropout, units=10) #  units = number of classes





    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    #    IN "PREIDCT" MODE, WE JUST NEED PREDICTIONS
    if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)   #   mode is always required
                                                                            #    predictions arg is required in PREDICT mode





    # Calculate Loss (for both TRAIN and EVAL modes)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=NUM_OF_CLASSES)   #make one-hot if needed

    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)      #change the optimizer if you want(ex:AdamOptimizer() )
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)  #   "train_op" arg is required in TRAIN mode






    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
    # in EVAL mode
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)    #   loss is required, but eval_metric_ops is optional





def main(unused_argv):

    # Load training and eval data

    train_data = FETCH_TRAIN_DATA #   (np.array())
    train_labels = FETCH_TRAIN_LABELS #   (np.array())

    eval_data = FETCH_EVAL_DATA #   (np.array())
    eval_labels = FETCH_EVAL_LABELS #   (np.array())




    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="./CHECKPOINTS")






    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    #list any others log (in this dict) you want to see during training.
    tensors_to_log = {"probabilities": "softmax_tensor"}  #   key name (here probabilities) is upto you, value is "name" arg of any layer

    logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)  #you will see the log after every 50 iterations






    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},  #features['x'] in cnn_model_fun input layer
      y=train_labels,
      batch_size=BATCH_SIZE,
      num_epochs=NUM_OF_EPOCHS,
      shuffle=True)

    mnist_classifier.train(
      input_fn=train_input_fn,
      steps=NUM_OF_STEPS,  #HOW MANY STEPS YOU WANT TO TRAIN FOR
      hooks=[logging_hook])     #   BIND THE HOOK YOU CREATED EARLIER




    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)

    #print(eval_results)







if __name__ == "__main__":
    tf.app.run()