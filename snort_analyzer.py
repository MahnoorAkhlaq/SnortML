import re
import pandas as pd
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("rnn_attack_classifier_model_updated.keras")

def parse_snort_log_entry(entry):
    pattern = r'(\d{2}/\d{2}-\d{2}:\d{2}:\d{2}\.\d{6}) \[\*\*\] \[\d+:\d+:\d+\] "(.*?)" \[\*\*\] \[Priority: (\d+)\] \{(\w+)\} (\d+\.\d+\.\d+\.\d+):(\d+) -> (\d+\.\d+\.\d+\.\d+):(\d+)'
    match = re.match(pattern, entry)
    if match:
        timestamp = match.group(1)
        message = match.group(2)
        priority = int(match.group(3))
        protocol = match.group(4)
        source_ip = match.group(5)
        source_port = int(match.group(6))
        dest_ip = match.group(7)
        dest_port = int(match.group(8))
        return timestamp, message, priority, protocol, source_ip, source_port, dest_ip, dest_port
    return None

def preprocess_log_entry(entry):    
    timestamp, message, priority, protocol, source_ip, source_port, dest_ip, dest_port = parse_snort_log_entry(entry)
    protocol_encoding = {'TCP': [1, 0, 0], 'UDP': [0, 1, 0], 'ICMP': [0, 0, 1]}
    protocol_encoded = protocol_encoding.get(protocol, [0, 0, 0])  
    priority_one_hot = [0] * 10  
    priority_one_hot[priority] = 1
    additional_features = [0] * (196 - len(protocol_encoded) - len(priority_one_hot)) 
    features = np.array(protocol_encoded + priority_one_hot + additional_features)
    features = np.reshape(features, (1, 1, 196))  # Reshape to (None, 1, 196)
    return timestamp, message, priority, protocol, source_ip, source_port, dest_ip, dest_port, features

def is_serious_attack(message):
    return "ping of death" in message.lower() or "ddos" in message.lower() or "tcp syn flood" in message.lower() or "udp" in message.lower()

def analyze_snort_log_file(file_path, model):
    with open(file_path, 'r') as file, open("/home/mahnoor/Desktop/Python/detected_attacks.csv", "w") as csv_file:
        csv_file.write("Timestamp,Message,Priority,Protocol,Source IP,Source Port,Destination IP,Destination Port,Label,Attack Seriousness\n")  # Write header
        for line in file:
            timestamp, message, priority, protocol, source_ip, source_port, dest_ip, dest_port, features = preprocess_log_entry(line.strip())
            prediction = model.predict(features)
            label = 0 if prediction > 0.5 or is_serious_attack(message) else 1  
            attack_seriousness = "SERIOUS" if label == 1 else "NOT SERIOUS"
            result_line = f"{timestamp},{message},{priority},{protocol},{source_ip},{source_port},{dest_ip},{dest_port},{label},{attack_seriousness}\n"
            csv_file.write(result_line)
            print("Attack Detected (Seriousness:", attack_seriousness + "):")
            print("Timestamp:", timestamp)
            print("Message:", message)
            print("Priority:", priority)
            print("Protocol:", protocol)
            print("Source IP:", source_ip)
            print("Source Port:", source_port)
            print("Destination IP:", dest_ip)
            print("Destination Port:", dest_port)
            print("Label:", label)
            print()

snort_log_file_path = "snort.log"
analyze_snort_log_file(snort_log_file_path, model)
