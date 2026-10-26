def penalized_loss(y_true, y_pred):

    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=len(class_name))
    
    base_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    
    misclassified_as_happy = tf.cast(
        tf.logical_and(
            tf.equal(y_true, angry_index), 
            tf.equal(tf.argmax(y_pred, axis=-1), happy_index) 
        ),
        tf.float32
    )

    penalty = 2.0 * misclassified_as_happy
 
    return base_loss + penalty