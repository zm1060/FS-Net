import tensorflow as tf
import numpy as np
from tqdm import tqdm
import json
import model
import os
import functools
from dataset import accuracy, get_dataset_from_generator
import eval


def train(config):
    max_len = config.max_flow_length_train
    with open(config.train_meta) as fp:
        train_num = int(fp.read().strip())
    with open(config.test_meta) as fp:
        dev_num = int(fp.read().strip())
    dev_ratio = config.eval_batch * config.batch_size / dev_num
    if config.eval_batch == -1:
        config.eval_batch = dev_num // config.batch_size + 1
        dev_ratio = 1
    train_dataset = get_dataset_from_generator(config.train_json, config, max_len)
    dev_dataset = get_dataset_from_generator(config.test_json, config, max_len, dev_ratio)

    if config.decay_step == 'auto':
        config.decay_step = train_num * 2 // config.batch_size + 1
    print('[Decay Step]:', config.decay_step)
    print('[Length Num]:', config.length_num)

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    train_iterator = train_dataset.make_one_shot_iterator()
    dev_app_iterator = dev_dataset.make_one_shot_iterator()
    rnn_classify = model.FSNet(config, iterator)

    for v in tf.trainable_variables():
        if v.shape.dims is None:
            print('%65s%5s' % (v.name, ' ' * 5), None)
        else:
            print('%65s%10d' % (v.name, functools.reduce(lambda x, y: x * y, v.shape)))

    sess_config = tf.ConfigProto(allow_soft_placement=True)

    loss_step = config.loss_save
    lr = config.learning_rate

    with tf.Session(config=sess_config) as sess:
        writer = tf.summary.FileWriter(config.log_dir)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables())

        train_handle = sess.run(train_iterator.string_handle())
        dev_app_handle = sess.run(dev_app_iterator.string_handle())

        sess.run(rnn_classify.train_false)
        sess.run(tf.assign(rnn_classify.lr, tf.constant(lr, dtype=tf.float32)))
        # writer.add_graph(sess.graph)

        for _ in tqdm(range(config.iter_num), ascii=True, desc='Training'):
            global_step = sess.run(rnn_classify.global_step) + 1

            loss, _, clr = sess.run([rnn_classify.loss, rnn_classify.train_op, rnn_classify.clr],
                                    feed_dict={handle: train_handle})
            if not (global_step % loss_step):  # save loss
                loss_sum = tf.Summary(value=[tf.Summary.Value(tag='model/loss', simple_value=loss)])
                writer.add_summary(loss_sum, global_step)

            if not (global_step % config.checkpoint):  # save model and compute train and test
                sess.run(rnn_classify.train_false)
                # compute train loss
                _, summary, metric = accuracy(rnn_classify, config.train_eval_batch, sess, handle, train_handle, 'train')
                tqdm.write('[Step={}] TRAIN batch: loss: {}, accuracy: {}'.format(
                    global_step, metric.get('train/loss/all'), metric.get('train/accuracy')))
                for s in summary:
                    writer.add_summary(s, global_step)
                # computer test loss
                loss_app, summary_app, metric = accuracy(rnn_classify, config.eval_batch, sess, handle, dev_app_handle, 'dev')
                tqdm.write('[Step={}] DEV batch: loss: {}, accuracy: {}'.format(
                    global_step, metric.get('dev/loss/all'), metric.get('dev/accuracy')))
                for s in summary_app:
                    writer.add_summary(s, global_step)
                sess.run(rnn_classify.train_true)

                lr_sum = tf.Summary(value=[tf.Summary.Value(tag='lr', simple_value=clr)])
                writer.add_summary(lr_sum, global_step)
                writer.flush()

                # save model
                saver.save(sess, os.path.join(config.model_dir, 'model_%d.ckpt' % global_step))
        writer.close()


def _predict_test(sess, model, num, class_num):
    pred = [[] for _ in range(class_num)]
    real = [[] for _ in range(class_num)]
    sample_set = set()
    for _ in tqdm(range(num), ascii=True, desc='Predict'):
        ids, preds = sess.run([model.ids, model.pred])
        for idx, predx in zip(ids.tolist(), preds.tolist()):
            idx = idx.decode('utf-8')
            if idx in sample_set:
                continue
            sample_set.add(idx)
            real_app = int(idx.strip().split('-')[0])
            real[real_app].append(real_app)
            pred[real_app].append(predx)
    return real, pred


def predict(config):
    test_dataset = get_dataset_from_generator(config.test_json, config, config.max_flow_length_test)
    test_dataset = test_dataset.make_one_shot_iterator()
    with open(config.test_meta) as fp:
        test_num = int(fp.read().strip())

    rnn_classify = model.FSNet(config, test_dataset, trainable=False)

    sess_config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(sess, tf.train.latest_checkpoint(config.test_model_dir))

        sess.run(rnn_classify.train_false)
        num = test_num // config.batch_size + 1

        real, pred = _predict_test(sess, rnn_classify, num, config.class_num)
        res = eval.evaluate(real, pred)
        eval.save_res(res, os.path.join(config.pred_dir, 'FSNet.json'))
        print(json.dumps(res, indent=1, sort_keys=True))
