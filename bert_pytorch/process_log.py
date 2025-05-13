import os
import pandas as pd
import numpy as np
from random import shuffle

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from drain3.file_persistence import FilePersistence

class Processor():
    def __init__(self, options):
      self.drain_config = options["drain_config"]
      self.drain_state = options["drain_state"]
      self.output_dir = options["output_dir"]
      self.seq_len = options["seq_len"]
      self.train_ratio = options["train_ratio"]

      self.structured_csv = os.path.join(self.output_dir, "process_structured.csv")

      self.drain_config = TemplateMinerConfig()
      self.drain_config.load(self.drain_config)

      self.drain_persistence = FilePersistence(self.drain_state)

      self.drain_miner = TemplateMiner(self.drain_persistence, self.drain_config) 

    def inference(self, msg):
        ID = self.drain_miner.add_log_message(msg)["cluster_id"]
        return ID

    def preprocess(self, path, labels=False):
        df = pd.read_csv(path)
        if not labels:
          df["Label"] = 0 # Assume that all message are not anomalous

        df["EventId"] = df.apply(lambda row: self.inference(row._value), axis=1)
        df["datetime"] = pd.to_datetime(df['_time'])
        df['timestamp'] = df["datetime"].values.astype(np.int64) // 10 ** 9
        
        df = df.sort_values(by=['timestamp']) # sort by timestamp

        df['deltaT'] = df['datetime'].diff() / np.timedelta64(1, 's')
        df['deltaT'].fillna(0)

        df = df[["timestamp", "Label", "EventId", "deltaT"]]

        df.to_csv(self.structured_csv, index=False)

    def process(self, shuffle_seq=True):
        df = pd.read_csv(self.structured_csv )

        events = df["EventId"].tolist()
        labels = df["Label"].tolist()

        sequences = []
        for i in range(len(events) - self.seq_len + 1):
            window = events[i:i + self.seq_len]
            sequences.append((window, max(labels[i:i + self.seq_len]))) #sequence plus label

        encoded_sequences = [seq for seq,_ in sequences]
        encoded_sequences = [" ".join(seq) for seq in encoded_sequences]

        encoded_sequences = [f"{sequences[i][0]}:" for i in range(len(encoded_sequences))]

        normal_sequences = [encoded_sequences[i] for i in range(len(encoded_sequences)) if sequences[i][1] == 0] 
        abnormal_sequences = [encoded_sequences[i] for i in range(len(encoded_sequences)) if sequences[i][1] == 1]
        normal_len = len(normal_sequences)

        if shuffle_seq:
          shuffle(normal_sequences)
          shuffle(abnormal_sequences)

        train_len = int(normal_len * self.train_ratio)

        train_sequence = normal_sequences[:train_len]
        test_sequence = normal_sequences[train_len:]

        with open(os.path.join(self.output_dir, "train"), "w") as f:
          for seq in train_sequence:
            f.write(seq + "\n")

        with open(os.path.join(self.output_dir, "test_normal"), "w") as f:
          for seq in test_sequence:
            f.write(seq + "\n")
        
        with open(os.path.join(self.output_dir, "test_abnormal"), "w") as f:
          for seq in abnormal_sequences:
            f.write(seq + "\n")
