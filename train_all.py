import os
import tensorflow as tf
import model
import train
from glob import glob
from tqdm import tqdm
import signal
import sys
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import argparse  # 添加参数解析

# 全局变量用于存储最佳性能
best_metrics = {
    'step': 0,
    'accuracy': 0.0,
    'precision': 0.0,
    'recall': 0.0,
    'f1': 0.0,
    'loss': float('inf')
}

def signal_handler(sig, frame):
    """处理Ctrl+C信号，优雅地退出并显示最佳性能"""
    print('\n\n' + '='*50)
    print('Training interrupted! Best performance on test set:')
    print(f'At step {best_metrics["step"]}:')
    print(f'Accuracy:  {best_metrics["accuracy"]:.4f}')
    print(f'Precision: {best_metrics["precision"]:.4f}')
    print(f'Recall:    {best_metrics["recall"]:.4f}')
    print(f'F1 Score:  {best_metrics["f1"]:.4f}')
    print(f'Loss:      {best_metrics["loss"]:.4f}')
    print('='*50)
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)

def calculate_metrics_from_metric(metric, prefix='train'):
    """从train.accuracy返回的metric计算指标"""
    return {
        'accuracy': metric[f"{prefix}/accuracy"],
        'precision': metric[f"{prefix}/precision"] if f"{prefix}/precision" in metric else 0.0,
        'recall': metric[f"{prefix}/recall"] if f"{prefix}/recall" in metric else 0.0,
        'f1': metric[f"{prefix}/f1"] if f"{prefix}/f1" in metric else 0.0
    }

def update_best_metrics(metrics, step, loss):
    """更新最佳指标"""
    global best_metrics
    # 如果是第一次更新，或者性能更好（F1更高或F1相同但损失更低）
    if (best_metrics['step'] == 0) or \
       (metrics['accuracy'] > best_metrics['accuracy']) or \
       (metrics['accuracy'] == best_metrics['accuracy'] and loss < best_metrics['loss']):
        best_metrics = {
            'step': step,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'loss': loss
        }

# 设置 GPU 内存增长
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# 设置使用的 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一个 GPU，如果有多个 GPU 可以改为 "0,1,2,3"

def get_num_classes(data_dir):
    """获取实际的类别数量"""
    with open(os.path.join(data_dir, 'status.label'), 'r') as f:
        return len(f.readlines())

def get_predictions_and_labels(rnn_classify, batch_num, sess, handle, data_handle):
    """获取模型预测结果和真实标签"""
    all_preds = []
    all_labels = []
    
    for _ in range(batch_num):
        # 使用正确的模型属性名称
        outputs = sess.run([rnn_classify.pred, rnn_classify.label],
                         feed_dict={handle: data_handle})
        predictions, labels = outputs
        all_preds.extend(predictions)
        all_labels.extend(labels)
    
    return np.array(all_labels), np.array(all_preds)

def calculate_metrics_with_predictions(labels, preds):
    """使用预测结果和真实标签计算所有指标"""
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='macro', zero_division=0),
        'recall': recall_score(labels, preds, average='macro', zero_division=0),
        'f1': f1_score(labels, preds, average='macro', zero_division=0)
    }

def custom_train(config):
    """自定义训练函数，添加进度条和指标显示"""
    global best_metrics
    # 重置最佳指标
    best_metrics = {
        'step': 0,
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'loss': float('inf')
    }
    
    with open(config.train_meta) as fp:
        train_num = int(fp.read().strip())
    with open(config.test_meta) as fp:
        dev_num = int(fp.read().strip())
    
    dev_ratio = config.eval_batch * config.batch_size / dev_num
    if config.eval_batch == -1:
        config.eval_batch = dev_num // config.batch_size + 1
        dev_ratio = 1
    
    print(f"Using {train_num} samples from {config.train_json}")
    
    train_dataset = train.get_dataset_from_generator(config.train_json, config, config.max_flow_length_train)
    dev_dataset = train.get_dataset_from_generator(config.test_json, config, config.max_flow_length_train, dev_ratio)

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    train_iterator = train_dataset.make_one_shot_iterator()
    dev_app_iterator = dev_dataset.make_one_shot_iterator()
    rnn_classify = model.FSNet(config, iterator)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    loss_step = config.loss_save
    lr = config.learning_rate

    with tf.Session(config=sess_config) as sess:
        # 确保日志目录存在
        if not os.path.exists(config.tensorboard_log_dir):
            os.makedirs(config.tensorboard_log_dir)
            print(f"Created TensorBoard log directory: {config.tensorboard_log_dir}")
        
        writer = tf.summary.FileWriter(config.tensorboard_log_dir)
        writer.add_graph(sess.graph)  # 添加计算图到日志
        
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables())

        train_handle = sess.run(train_iterator.string_handle())
        dev_app_handle = sess.run(dev_app_iterator.string_handle())

        sess.run(rnn_classify.train_false)
        sess.run(tf.assign(rnn_classify.lr, tf.constant(lr, dtype=tf.float32)))

        # 使用tqdm显示训练进度
        pbar = tqdm(range(config.iter_num), desc='Training')
        try:
            for _ in pbar:
                global_step = sess.run(rnn_classify.global_step) + 1

                # 训练一步
                loss, c_loss, r_loss, _, clr = sess.run(
                    [rnn_classify.loss, rnn_classify.c_loss, rnn_classify.rec_loss, 
                     rnn_classify.train_op, rnn_classify.clr],
                    feed_dict={handle: train_handle})

                # 更新进度条信息
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'clf_loss': f'{c_loss:.4f}',
                    'rec_loss': f'{r_loss:.4f}',
                    'lr': f'{clr:.6f}'
                })

                # 每10步计算一次训练和验证指标
                if not (global_step % 10):
                    sess.run(rnn_classify.train_false)
                    
                    # 计算训练指标
                    y_true, y_pred = get_predictions_and_labels(
                        rnn_classify, 1, sess, handle, train_handle)
                    train_metrics = calculate_metrics_with_predictions(y_true, y_pred)
                    
                    print(f'\n[Step={global_step}] TRAIN: loss={loss:.4f}, '
                          f'accuracy={train_metrics["accuracy"]:.4f}, '
                          f'F1={train_metrics["f1"]:.4f}')
                    
                    # 计算验证指标
                    y_true, y_pred = get_predictions_and_labels(
                        rnn_classify, 1, sess, handle, dev_app_handle)
                    dev_metrics = calculate_metrics_with_predictions(y_true, y_pred)
                    
                    # 更新最佳指标
                    update_best_metrics(dev_metrics, global_step, loss)
                    
                    print(f'[Step={global_step}] DEV: loss={loss:.4f}, '
                          f'accuracy={dev_metrics["accuracy"]:.4f}, '
                          f'F1={dev_metrics["f1"]:.4f}\n')
                    
                    sess.run(rnn_classify.train_true)

                if not (global_step % loss_step):  # 保存loss
                    loss_sum = tf.Summary(value=[tf.Summary.Value(tag='model/loss', simple_value=loss)])
                    writer.add_summary(loss_sum, global_step)

                if not (global_step % config.checkpoint):  # 完整评估和保存模型
                    sess.run(rnn_classify.train_false)
                    
                    # 完整计算训练指标
                    y_true, y_pred = get_predictions_and_labels(
                        rnn_classify, config.train_eval_batch, sess, handle, train_handle)
                    train_metrics = calculate_metrics_with_predictions(y_true, y_pred)
                    
                    print(f'\n[Full Eval Step={global_step}] TRAIN: loss={loss:.4f}, '
                          f'accuracy={train_metrics["accuracy"]:.4f}, '
                          f'F1={train_metrics["f1"]:.4f}')
                    
                    # 完整计算验证指标
                    y_true, y_pred = get_predictions_and_labels(
                        rnn_classify, config.eval_batch, sess, handle, dev_app_handle)
                    dev_metrics = calculate_metrics_with_predictions(y_true, y_pred)
                    
                    # 更新最佳指标
                    update_best_metrics(dev_metrics, global_step, loss)
                    
                    # 打印当前详细指标
                    print(f'\n[Step={global_step}] TEST Metrics:')
                    print(f'Accuracy:  {dev_metrics["accuracy"]:.4f}')
                    print(f'Precision: {dev_metrics["precision"]:.4f}')
                    print(f'Recall:    {dev_metrics["recall"]:.4f}')
                    print(f'F1 Score:  {dev_metrics["f1"]:.4f}')
                    print(f'Loss:      {loss:.4f}\n')
                    
                    sess.run(rnn_classify.train_true)

                    lr_sum = tf.Summary(value=[tf.Summary.Value(tag='lr', simple_value=clr)])
                    writer.add_summary(lr_sum, global_step)
                    writer.flush()

                    # 保存模型
                    saver.save(sess, os.path.join(config.model_dir, f'model_{global_step}.ckpt'))

                # 每10步记录详细的训练指标
                if not (global_step % 10):
                    # 创建摘要
                    summaries = [
                        tf.Summary.Value(tag='metrics/loss', simple_value=loss),
                        tf.Summary.Value(tag='metrics/clf_loss', simple_value=c_loss),
                        tf.Summary.Value(tag='metrics/rec_loss', simple_value=r_loss),
                        tf.Summary.Value(tag='metrics/learning_rate', simple_value=clr)
                    ]
                    summary = tf.Summary(value=summaries)
                    writer.add_summary(summary, global_step)
                    writer.flush()  # 确保立即写入

        except KeyboardInterrupt:
            # 处理Ctrl+C中断
            signal_handler(signal.SIGINT, None)
        
        # 训练结束时显示最佳性能
        print('\n' + '='*50)
        print('Training completed! Best performance on test set:')
        print(f'At step {best_metrics["step"]}:')
        print(f'Accuracy:  {best_metrics["accuracy"]:.4f}')
        print(f'Precision: {best_metrics["precision"]:.4f}')
        print(f'Recall:    {best_metrics["recall"]:.4f}')
        print(f'F1 Score:  {best_metrics["f1"]:.4f}')
        print(f'Loss:      {best_metrics["loss"]:.4f}')
        print('='*50)
        
        writer.close()

def train_single_dataset(dataset_dir):
    """训练单个数据集"""
    dataset_name = os.path.basename(dataset_dir)
    print(f"\nTraining dataset: {dataset_name}")
    
    # 设置目录
    record_dir = os.path.join(dataset_dir, 'record')
    log_base = os.path.join('log', dataset_name)
    model_dir = os.path.join(log_base, 'checkpoints')
    tensorboard_dir = os.path.join(log_base, 'tensorboard')
    data_dir = os.path.join(dataset_dir, 'filter')
    pred_dir = os.path.join('result', dataset_name)
    
    # 创建必要的目录
    for dirx in [log_base, model_dir, tensorboard_dir, pred_dir]:
        if not os.path.exists(dirx):
            os.makedirs(dirx)
            print(f"Created directory: {dirx}")
    
    # 设置配置
    flags = tf.flags
    FLAGS = flags.FLAGS
    
    # 设置文件路径
    flags.DEFINE_string('train_json', os.path.join(record_dir, 'train.json'), 'train json file')
    flags.DEFINE_string('test_json', os.path.join(record_dir, 'test.json'), 'test json file')
    flags.DEFINE_string('train_meta', os.path.join(record_dir, 'train.meta'), 'train meta file')
    flags.DEFINE_string('test_meta', os.path.join(record_dir, 'test.meta'), 'test meta file')
    flags.DEFINE_string('model_dir', model_dir, 'model directory')
    flags.DEFINE_string('tensorboard_log_dir', tensorboard_dir, 'tensorboard log directory')
    flags.DEFINE_string('data_dir', data_dir, 'data directory')
    flags.DEFINE_string('pred_dir', pred_dir, 'prediction directory')
    
    # 获取实际的类别数量
    num_classes = get_num_classes(data_dir)
    print(f"Number of classes: {num_classes}")
    
    # 模型参数
    flags.DEFINE_integer('class_num', num_classes, 'number of classes')
    flags.DEFINE_integer('length_block', 1, 'length of a block')
    flags.DEFINE_integer('min_length', 2, 'minimum flow length')
    flags.DEFINE_integer('max_packet_length', 5000, 'maximum packet length')
    flags.DEFINE_float('split_ratio', 0.8, 'train/test split ratio')
    flags.DEFINE_float('keep_ratio', 1, 'ratio of keeping examples')
    flags.DEFINE_integer('max_flow_length_train', 200, 'maximum flow length for training')
    flags.DEFINE_integer('max_flow_length_test', 2000, 'maximum flow length for testing')
    
    # 计算length_num
    length_num = FLAGS.max_packet_length // FLAGS.length_block + 4
    
    # 训练参数
    flags.DEFINE_integer('batch_size', 128, 'batch size')
    flags.DEFINE_integer('hidden', 128, 'hidden size')
    flags.DEFINE_integer('layer', 2, 'number of layers')
    flags.DEFINE_integer('length_dim', 16, 'dimension of length embedding')
    flags.DEFINE_integer('length_num', length_num, 'length_num')
    flags.DEFINE_float('keep_prob', 0.8, 'keep probability')
    flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
    flags.DEFINE_integer('iter_num', int(1.2e5), 'number of iterations')
    flags.DEFINE_integer('eval_batch', 77, 'evaluation batch size')
    flags.DEFINE_integer('train_eval_batch', 77, 'train evaluation batch size')
    
    # 计算decay_step
    with open(FLAGS.train_meta) as fp:
        train_num = int(fp.read().strip())
    decay_step = train_num * 2 // FLAGS.batch_size + 1
    
    flags.DEFINE_integer('decay_step', decay_step, 'decay step')
    flags.DEFINE_float('decay_rate', 0.5, 'decay rate')
    flags.DEFINE_string('mode', 'train', 'train or test mode')
    flags.DEFINE_integer('capacity', int(1e3), 'dataset shuffle capacity')
    flags.DEFINE_integer('loss_save', 100, 'steps to save loss')
    flags.DEFINE_integer('checkpoint', 5000, 'steps to save checkpoint')
    flags.DEFINE_float('grad_clip', 5.0, 'gradient clipping')
    flags.DEFINE_boolean('is_cudnn', False, 'whether to use cudnn')
    flags.DEFINE_float('rec_loss', 0.5, 'reconstruction loss weight')
    
    # 训练模型
    print("\nStarting training...")
    print(f"Logs will be saved to: {tensorboard_dir}")
    print(f"Models will be saved to: {model_dir}")
    custom_train(FLAGS)

def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='Train network traffic classifier')
    parser.add_argument('--dataset', type=str, help='指定要训练的数据集名称。如果不指定，则训练所有数据集')
    parser.add_argument('--processed_dir', type=str, default='./processed_datasets', 
                       help='处理后的数据集目录')
    args = parser.parse_args()
    
    # 获取所有处理好的数据集目录
    processed_datasets = glob(os.path.join(args.processed_dir, '*'))
    
    if args.dataset:
        dataset_path = os.path.join(args.processed_dir, args.dataset)
        if not os.path.exists(dataset_path):
            print(f"错误：找不到指定的已处理数据集 '{args.dataset}'")
            print(f"可用的已处理数据集: {', '.join(os.path.basename(d) for d in processed_datasets)}")
            return
        processed_datasets = [dataset_path]
    
    print(f"将训练以下数据集: {', '.join(os.path.basename(d) for d in processed_datasets)}")
    
    # 训练每个选定的数据集
    for dataset_dir in processed_datasets:
        print(f"\nProcessing dataset: {os.path.basename(dataset_dir)}")
        
        # 重置 FLAGS
        tf.flags.FLAGS.unparse_flags()
        tf.flags.FLAGS.mark_as_parsed()
        
        # 训练单个数据集
        train_single_dataset(dataset_dir)

if __name__ == '__main__':
    main() 