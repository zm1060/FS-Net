import random
import json
import os
from scapy.all import rdpcap
from scapy.layers.inet import IP
import glob
from scapy.utils import PcapReader

def extract_packet_lengths(pcap_file):
    """从pcap文件中提取数据包长度序列"""
    try:
        # 使用 PcapReader 而不是 rdpcap，以更好地处理大文件和损坏的文件
        packet_lengths = []
        with PcapReader(pcap_file) as pcap_reader:
            for packet in pcap_reader:
                if IP in packet:
                    # 获取IP包的总长度
                    packet_lengths.append(len(packet))
                    
                    if len(packet_lengths) >= 100:  # 限制最大包数
                        break
                        
        # 确保至少有2个包
        if len(packet_lengths) < 2:
            print(f"Warning: {pcap_file} contains less than 2 packets")
            return None
            
        print(f"Extracted {len(packet_lengths)} packets from {pcap_file}")
        return packet_lengths
    
    except Exception as e:
        print(f"Error processing {pcap_file}: {str(e)}")
        return None

def process_flow(original_lengths):
    """处理流量特征"""
    length_block = 500
    return [length // length_block + 3 for length in original_lengths]

def generate_dataset(pcap_dir):
    """从pcap文件生成数据集"""
    dataset = []
    
    # 获取所有pcap文件
    pcap_files = glob.glob(os.path.join(pcap_dir, "*.pcap"))
    print(f"Found {len(pcap_files)} pcap files")
    
    # 创建标签映射
    label_to_id = create_label_mapping(pcap_files)
    
    for pcap_file in pcap_files:
        try:
            # 使用文件名（不含扩展名）作为标签
            app_label = os.path.splitext(os.path.basename(pcap_file))[0]
            label_id = label_to_id[app_label]  # 获取标签对应的整数ID
            print(f"Processing {app_label} (ID: {label_id})...")
            
            # 提取特征
            packet_lengths = extract_packet_lengths(pcap_file)
            if packet_lengths is None or len(packet_lengths) < 2:
                print(f"Skipping {pcap_file} due to insufficient packets")
                continue
                
            # 处理特征
            processed_flow = process_flow(packet_lengths)
            
            # 构建数据样本
            sample = {
                "label": label_id,  # 使用整数标签ID
                "flow": processed_flow,
                "lo": packet_lengths,
                "id": app_label
            }
            
            dataset.append(sample)
            
            # 生成原始.num文件内容
            length_str = '\t'.join(map(str, packet_lengths))
            original_format = f";{length_str};\n"
            
            # 保存到对应的.num文件
            with open(f"origin_data/{app_label}.num", "w") as f:
                f.write(original_format)
                
        except Exception as e:
            print(f"Error processing {pcap_file}: {e}")
            continue
    
    if len(dataset) == 0:
        print("Warning: No valid samples were generated!")
    else:
        print(f"Generated {len(dataset)} samples in total")
    
    return dataset

def create_label_mapping(pcap_files):
    """创建标签到整数的映射"""
    unique_labels = sorted(set(os.path.splitext(os.path.basename(f))[0] for f in pcap_files))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    
    # 保存标签映射
    with open("filter/status.label", "w") as f:
        for label, idx in label_to_id.items():
            f.write(f"{label}\t{idx}\n")
    
    return label_to_id

def main():
    # 创建必要的目录
    os.makedirs("origin_data", exist_ok=True)
    os.makedirs("record", exist_ok=True)
    os.makedirs("filter", exist_ok=True)
    
    # 清空原有的.num文件
    for f in glob.glob("origin_data/*.num"):
        os.remove(f)
    
    # 从pcap文件生成数据集
    dataset = generate_dataset("./")
    
    if not dataset:
        print("未找到有效的pcap文件或提取失败！")
        return
    
    # 随机划分训练集和测试集
    random.shuffle(dataset)
    split_point = int(len(dataset) * 0.8)
    train_data = dataset[:split_point]
    test_data = dataset[split_point:]
    
    # 保存为JSON格式
    with open("record/train.json", "w") as f:
        json.dump(train_data, f, indent=2)
    
    with open("record/test.json", "w") as f:
        json.dump(test_data, f, indent=2)
    
    # 保存数据集大小信息
    with open("record/train.meta", "w") as f:
        f.write(str(len(train_data)))
    
    with open("record/test.meta", "w") as f:
        f.write(str(len(test_data)))

if __name__ == "__main__":
    main()