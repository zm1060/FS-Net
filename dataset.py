import tensorflow as tf
from tqdm import tqdm
import numpy as np
import json


PAD_KEY = 0
START_KEY = 1
END_KEY = 2


def read_file_generator(filename, max_len, keep_ratio=1):
    def gen():
        try:
            with open(filename) as fp:
                data = json.load(fp)
                
            # 过滤并处理数据
            data_all = []
            for exp in data:
                flow_length = len(exp['flow'])
                # 修改长度判断逻辑
                if flow_length > 0:  # 只要有数据就接受
                    # 如果超过max_len就截断，否则填充
                    if flow_length > max_len:
                        flow = [START_KEY] + exp['flow'][:max_len] + [END_KEY]
                    else:
                        flow = [START_KEY] + exp['flow'] + [END_KEY] + [PAD_KEY] * (max_len - flow_length)
                    data_all.append((str.encode(exp['id']), exp['label'], flow))
            
            # 检查是否有有效数据
            if not data_all:
                print(f"Warning: No valid data found in {filename}")
                return
                
            # 应用 keep_ratio
            total_num = max(1, int(keep_ratio * len(data_all)))  # 至少保留1个样本
            data_all = data_all[:total_num]
            print(f"Using {len(data_all)} samples from {filename}")
            
            numx = 0
            while True:
                if numx == 0:
                    np.random.shuffle(data_all)
                yield data_all[numx]
                numx = (numx + 1) % len(data_all)
                
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return
            
    return gen


def get_dataset_from_generator(file, config, max_len, keep_ratio=1):
    data_gen = read_file_generator(file, max_len, keep_ratio)
    
    # 创建数据集
    dataset = tf.data.Dataset.from_generator(
        data_gen,
        (tf.string, tf.int32, tf.int32),
        (tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([max_len + 2]))
    )
    
    # 添加数据处理
    dataset = dataset.repeat()  # 无限重复数据集
    dataset = dataset.shuffle(config.capacity)
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(4)
    
    return dataset


def _get_summary(metric):
    summ = []
    for met in metric:
        sx = tf.Summary(value=[tf.Summary.Value(tag=met, simple_value=metric[met])])
        summ.append(sx)
    return summ


def accuracy(model, val_num_batches, sess, handle, str_handle, name):
    pred_all, pred_right, losses, r_losses, c_losses = 0, 0, [], [], []
    metric = {}
    
    for _ in tqdm(range(val_num_batches), desc='eval', ascii=True):
        loss, c_loss, r_loss, pred, label = sess.run(
            [model.loss, model.c_loss, model.rec_loss, model.pred, model.label],
            feed_dict={handle: str_handle}
        )
        
        losses.append(loss)
        r_losses.append(r_loss)
        c_losses.append(c_loss)
        pred_all += len(pred)
        pred_right += np.sum(pred == label)
        
    # 计算指标
    loss = np.mean(losses)
    metric[name + '/loss/all'] = loss
    metric[name + '/loss/clf'] = np.mean(c_losses)
    metric[name + '/loss/rec'] = np.mean(r_losses)
    metric[name + '/accuracy'] = pred_right / pred_all if pred_all > 0 else 0
    
    summ = _get_summary(metric)
    return loss, summ, metric