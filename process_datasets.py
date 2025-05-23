import os
import glob
import json
from scapy.utils import PcapReader
from scapy.layers.inet import IP, TCP, UDP
import random
from tqdm import tqdm
import shutil
from collections import defaultdict
import argparse  # 添加参数解析

def extract_flows(pcap_file, max_packets_per_flow=100, min_packets=2):
    """从pcap文件中提取不同的流"""
    flows = defaultdict(list)
    try:
        with PcapReader(pcap_file) as pcap_reader:
            for packet in pcap_reader:
                if IP in packet and (TCP in packet or UDP in packet):
                    # 提取IP和端口信息
                    ip_src = packet[IP].src
                    ip_dst = packet[IP].dst
                    
                    if TCP in packet:
                        proto = 'TCP'
                        sport = packet[TCP].sport
                        dport = packet[TCP].dport
                    else:  # UDP
                        proto = 'UDP'
                        sport = packet[UDP].sport
                        dport = packet[UDP].dport
                    
                    # 创建双向流标识符（保持固定顺序）
                    if f"{ip_src}:{sport}" < f"{ip_dst}:{dport}":
                        flow_id = f"{ip_src}:{sport}-{ip_dst}:{dport}-{proto}"
                    else:
                        flow_id = f"{ip_dst}:{dport}-{ip_src}:{sport}-{proto}"
                    
                    # 只保存包长度
                    flows[flow_id].append(len(packet))
                    
                    # 如果某个流的包数达到上限，停止该流的收集
                    if len(flows[flow_id]) >= max_packets_per_flow:
                        continue
        
        # 过滤掉包数太少的流
        valid_flows = {k: v for k, v in flows.items() if len(v) >= min_packets}
        
        if not valid_flows:
            print(f"Warning: No valid flows found in {pcap_file}")
            return None
            
        return valid_flows
    
    except Exception as e:
        print(f"Error processing {pcap_file}: {str(e)}")
        return None

def process_flow(original_lengths, length_block=500):
    """处理流量特征，与原始实现保持一致"""
    return [length // length_block + 3 for length in original_lengths]

def process_single_dataset(dataset_path, output_base_dir, max_packets=256, min_packets=5):
    """处理单个数据集目录"""
    dataset_name = os.path.basename(dataset_path)
    print(f"\nProcessing dataset: {dataset_name}")
    
    # 创建输出目录
    dataset_output_dir = os.path.join(output_base_dir, dataset_name)
    os.makedirs(os.path.join(dataset_output_dir, "origin_data"), exist_ok=True)
    os.makedirs(os.path.join(dataset_output_dir, "record"), exist_ok=True)
    os.makedirs(os.path.join(dataset_output_dir, "filter"), exist_ok=True)
    
    # 获取所有类别目录
    class_dirs = [d for d in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, d))]
    
    # 创建标签映射
    label_to_id = {label: idx for idx, label in enumerate(sorted(class_dirs))}
    
    # 保存标签映射
    with open(os.path.join(dataset_output_dir, "filter/status.label"), "w") as f:
        for label, idx in label_to_id.items():
            f.write(f"{label}\t{idx}\n")
    
    dataset = []
    
    # 处理每个类别
    for class_name in tqdm(class_dirs, desc="Processing classes"):
        class_path = os.path.join(dataset_path, class_name)
        label_id = label_to_id[class_name]
        
        # 获取该类别下的所有pcap文件
        pcap_files = glob.glob(os.path.join(class_path, "*.pcap"))
        
        # 用于存储该类别的所有流
        class_flows = []
        flow_count = 0
        
        for pcap_file in tqdm(pcap_files, desc=f"Processing {class_name} pcaps", leave=False):
            # 提取所有流
            flows = extract_flows(pcap_file, max_packets, min_packets)
            if flows is None:
                continue
            
            # 处理每个流
            for flow_id, packet_lengths in flows.items():
                # 处理特征
                processed_flow = process_flow(packet_lengths)
                
                # 构建数据样本（与原始实现格式保持一致）
                sample = {
                    "label": label_id,
                    "flow": processed_flow,
                    "lo": packet_lengths,
                    "id": f"{class_name}-{flow_id}-{flow_count}"  # 包含流ID以确保唯一性
                }
                
                dataset.append(sample)
                class_flows.append(packet_lengths)
                flow_count += 1
        
        # 为每个类别生成.num文件（与原始实现格式保持一致）
        if class_flows:
            num_file = os.path.join(dataset_output_dir, "origin_data", f"{class_name}.num")
            with open(num_file, "w") as f:
                for flow in class_flows:
                    length_str = '\t'.join(map(str, flow))
                    f.write(f";{length_str};\n")
    
    if len(dataset) == 0:
        print(f"Warning: No valid samples were generated for dataset {dataset_name}!")
        return
    
    print(f"Generated {len(dataset)} flows from dataset {dataset_name}")
    
    # 随机划分训练集和测试集
    random.shuffle(dataset)
    split_point = int(len(dataset) * 0.8)
    train_data = dataset[:split_point]
    test_data = dataset[split_point:]
    
    # 保存为JSON格式
    with open(os.path.join(dataset_output_dir, "record/train.json"), "w") as f:
        json.dump(train_data, f, indent=2)
    
    with open(os.path.join(dataset_output_dir, "record/test.json"), "w") as f:
        json.dump(test_data, f, indent=2)
    
    # 保存数据集大小信息
    with open(os.path.join(dataset_output_dir, "record/train.meta"), "w") as f:
        f.write(str(len(train_data)))
    
    with open(os.path.join(dataset_output_dir, "record/test.meta"), "w") as f:
        f.write(str(len(test_data)))

def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='Process network traffic datasets')
    parser.add_argument('--dataset', type=str, help='指定要处理的数据集名称。如果不指定，则处理所有数据集')
    parser.add_argument('--input_dir', type=str, default='./datasets', help='输入数据集的基础目录')
    parser.add_argument('--output_dir', type=str, default='./processed_datasets', help='处理后的输出目录')
    parser.add_argument('--max_packets', type=int, default=256, help='每个流最大的包数量')
    parser.add_argument('--min_packets', type=int, default=5, help='每个流最小的包数量')
    args = parser.parse_args()
    
    # 创建输出基础目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取所有数据集目录
    dataset_dirs = [d for d in os.listdir(args.input_dir) 
                   if os.path.isdir(os.path.join(args.input_dir, d))]
    
    if args.dataset:
        if args.dataset not in dataset_dirs:
            print(f"错误：找不到指定的数据集 '{args.dataset}'")
            print(f"可用的数据集: {', '.join(dataset_dirs)}")
            return
        dataset_dirs = [args.dataset]
    
    print(f"将处理以下数据集: {', '.join(dataset_dirs)}")
    
    # 处理选定的数据集
    for dataset_dir in dataset_dirs:
        dataset_path = os.path.join(args.input_dir, dataset_dir)
        process_single_dataset(dataset_path, args.output_dir, 
                             max_packets=args.max_packets,
                             min_packets=args.min_packets)

if __name__ == "__main__":
    main() 