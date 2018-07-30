from __future__ import division

import tensorflow as tf
import json
import os
import math

tf.logging.set_verbosity(tf.logging.INFO)


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    name='model_dir', default='./models',
    help='Output directory for model and training stats.')
tf.app.flags.DEFINE_string(
    name='train_metadata', default=None,
    help='Regex that matches the training tf-records')
tf.app.flags.DEFINE_string(
    name='eval_metadata', default=None,
    help='Regex that matches the evaluation tf-records')
tf.app.flags.DEFINE_integer(
    name='total_steps', default=1, help='Steps to train')
tf.app.flags.DEFINE_integer(
    name='batch_size', default=1, help='Batch size')
tf.app.flags.DEFINE_float(
    name='learning_rate', default=1e-3, help='Learning rate')
tf.app.flags.DEFINE_float(
    name='reg_val', default=1e-4, help='Regularization loss')
tf.app.flags.DEFINE_integer(
    name='batch_prefetch', default=1, help='Batch prefetch')

TRAIN_SET_SIZE = 3200
EVAL_SET_SIZE = 800
IMAGE_SIZE = 101


def model_fn():
    def _model_fn(features, labels, mode, params):

        batch_image = tf.map_fn(preproc,features['image'],tf.float32)
        tf.logging.info("Input batch shape: {}".format(batch_image.get_shape()))

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        def _conv(_input, n_filters, kernel_size, reg_val):
            return tf.layers.conv2d(_input, filters=n_filters,
                                    kernel_size=kernel_size,
                                    strides=(1, 1),
                                    activation=tf.nn.elu,
                                    padding='same',
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                        reg_val),
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                    bias_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                    bias_regularizer=tf.contrib.layers.l2_regularizer(reg_val))

        with tf.variable_scope("TGS"):
            net = _conv(batch_image, 16, (3, 3), FLAGS.reg_val)
            net = _conv(net, 32, (3, 3), FLAGS.reg_val)
            net = _conv(net, 64, (3, 3), FLAGS.reg_val)
            net = _conv(net, 32, (3, 3), FLAGS.reg_val)
            net = _conv(net, 16, (3, 3), FLAGS.reg_val)
            net = _conv(net, 2, (3, 3), FLAGS.reg_val)
            net = _conv(net, 2, (1, 1), FLAGS.reg_val)
            #net = tf.sigmoid(net)

        predicted_mask = net
        prediction_dict = {"predicted_mask": predicted_mask}

        # Loss, training and eval operations are not needed during inference.
        total_loss = None
        loss = None
        train_op = None
        eval_metric_ops = {}
        export_outputs = None

        if mode != tf.estimator.ModeKeys.PREDICT:
            
            batch_mask = tf.map_fn(preproc,labels['mask'],tf.float32)
            tf.logging.info("Mask batch shape: {}".format(batch_mask.get_shape()))
            
            batch_neg = 1+tf.negative(batch_mask)            
            batch_mask_plus_neg = tf.squeeze(tf.stack([batch_mask,batch_neg],axis=3),axis=4)
            
            # IT IS VERY IMPORTANT TO RETRIEVE THE REGULARIZATION LOSSES
            reg_loss = tf.losses.get_regularization_loss()

            # This summary is automatically caught by the Estimator API
            tf.summary.scalar("Regularization_Loss", tensor=reg_loss)

            reshape_net = net
            reshape_mask = batch_mask_plus_neg            
            loss = tf.losses.softmax_cross_entropy(onehot_labels=reshape_mask,logits=reshape_net)
        
            tf.summary.scalar("XEntropy_LOSS", tensor=loss)

            total_loss = loss + reg_loss

            learning_rate = tf.constant(
                FLAGS.learning_rate, name='fixed_learning_rate')
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            vars_to_train = tf.trainable_variables()
            tf.logging.info("Variables to train: {}".format(vars_to_train))            

            if is_training:
                # You DO must get this collection in order to perform updates on batch_norm variables
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.minimize(
                        loss=total_loss, global_step=tf.train.get_global_step(), var_list=vars_to_train)

            eval_metric_ops = metrics(batch_mask_plus_neg, predicted_mask)

        else:            
            export_outputs = {'predicted_mask': tf.estimator.export.PredictOutput(
                outputs=predicted_mask)}
            #export_outputs = None
            

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=prediction_dict,
            loss=total_loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            export_outputs=export_outputs)

    return _model_fn


def preproc(image_bytes):
    image_png = tf.to_float(tf.image.decode_png(
        image_bytes, channels=1)) / 255.0
    image_png = tf.reshape(image_png, [IMAGE_SIZE, IMAGE_SIZE, 1],name="Reshape_Preproc")

    return image_png


def input_fn(metadata, batch_prefetch, batch_size, epochs):

    def _parse_function(example_proto):
        """ Parse data from tf.Example. All labels are decoded as float32"""

        parser_dict = {
            "image_bytes": tf.FixedLenFeature((), tf.string, default_value=""),
            "mask_bytes": tf.FixedLenFeature((), tf.string, default_value="")
        }
        parsed_features = tf.parse_single_example(example_proto, parser_dict)

        return parsed_features

    def _decode_images(features):
        
        return {'image': features['image_bytes']}, {'mask': features['mask_bytes']}

    def _input_fn():
        with tf.name_scope('Data_Loader'):

            dataset = tf.data.TFRecordDataset(
                metadata, compression_type="GZIP")
            dataset = dataset.map(_parse_function)
            dataset = dataset.map(_decode_images)
            dataset = dataset.shuffle(buffer_size=batch_prefetch * batch_size)

            dataset = dataset.repeat(epochs)
            return dataset.batch(batch_size)

    return _input_fn


def metrics(predicted_mask, target_mask):

    round_predicted_mask = tf.argmax(predicted_mask,axis=3)
    round_target_mask = tf.argmax(target_mask,axis=3)

    tf.logging.info("SHAPE {} \n\n".format(round_predicted_mask.get_shape()))

    mask_sum = round_predicted_mask + round_target_mask
    intersection = tf.reduce_sum(tf.cast(tf.equal(mask_sum, 2), tf.float32))
    union = tf.reduce_sum(tf.cast(tf.greater_equal(mask_sum, 1), tf.float32))
    _iou = (intersection / union)/FLAGS.batch_size
    iou_val = tf.get_variable('iou', trainable=False, validate_shape=True,
                              dtype=tf.float32, initializer=tf.constant(0, dtype=tf.float32))
    iou_op = iou_val.assign(iou_val + _iou)

    return {'IoU': (tf.identity(iou_val), iou_op)}


def get_serving_fn():
    return tf.estimator.export.build_raw_serving_input_receiver_fn({'image': tf.placeholder(dtype=tf.string, shape=[None])})


def list_tfrecord(regex):
    list_op = tf.train.match_filenames_once(regex)
    init_ops = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_ops)
        files = sess.run(list_op)

    return files


def train_tgs():

    train_metadata = list_tfrecord(FLAGS.train_metadata)
    eval_metadata = list_tfrecord(FLAGS.eval_metadata)

    epochs = int(math.ceil(FLAGS.total_steps /
                           (TRAIN_SET_SIZE / FLAGS.batch_size)))

    env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    task_data = env.get('task') or {'type': 'master', 'index': 0}
    trial = task_data.get('trial')

    if trial is not None:
        output_dir = os.path.join(FLAGS.model_dir, trial)
        tf.logging.info(
            "Hyperparameter Tuning - Trial {}. model_dir = {}".format(trial, output_dir))
    else:
        output_dir = FLAGS.model_dir

    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )
    #session_config = None

    run_config = tf.estimator.RunConfig(
        model_dir=output_dir,
        save_summary_steps=100,
        session_config=session_config,
        save_checkpoints_steps=1000,
        save_checkpoints_secs=None,
        keep_checkpoint_max=5
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn(),
        config=run_config
    )

    train_input_fn = input_fn(
        batch_size=FLAGS.batch_size, metadata=train_metadata, epochs=epochs, batch_prefetch=FLAGS.batch_prefetch)
    eval_input_fn = input_fn(
        batch_size=FLAGS.batch_size, metadata=eval_metadata, epochs=1, batch_prefetch=FLAGS.batch_prefetch)

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=FLAGS.total_steps)

    eval_steps = math.ceil(EVAL_SET_SIZE / FLAGS.batch_size)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=eval_steps,
        start_delay_secs=30,
        throttle_secs=300)

    tf.estimator.train_and_evaluate(
        estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)

    estimator.export_savedmodel(
        export_dir_base=output_dir, serving_input_receiver_fn=get_serving_fn())


if __name__ == "__main__":
    train_tgs()
