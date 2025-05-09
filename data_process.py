import sys
sys.path.append('../')

import os
import re
import json
import pandas as pd
from collections import defaultdict
import gc
from tqdm import tqdm
import numpy as np
from collections import Counter
from bert_pytorch.dataset.session import sliding_window
from random import shuffle

# get [log key, delta time] as input for deeplog
# input_dir  = os.path.expanduser('~/.dataset/hdfs/')
# output_dir = '../output/hdfs/'  # The output directory of parsing results
# log_file   = "HDFS.log"  # The input log file name

# log_structured_file = output_dir + log_file + "_structured.csv"
# log_templates_file = output_dir + log_file + "_templates.csv"
# log_sequence_file = output_dir + "hdfs_sequence.csv"

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from drain3.file_persistence import FilePersistence

# def file_generator(filename, df, features):
#     with open(filename, 'w') as f:
#         for _, row in df.iterrows():
#             for val in zip(*row[features]):
#                 f.write(','.join([str(v) for v in val]) + ' ')
#             f.write('\n')

def init_miner(config_path, state_path):
    config = TemplateMinerConfig()
    config.load(config_path)

    persistence = FilePersistence(state_path)
    template_miner = TemplateMiner(persistence, config) #load state

    return template_miner

def inference(msg, template_miner):
    ID = template_miner.add_log_message(msg)["cluster_id"]
    return f"E{ID}"

def preprocess(in_csv, out_csv, miner_config, miner_state):
    """
    Use Drain to map messages to events.

    inpath : Path to a CSV file with log messages
    outpath : Path to where preprocessed log messages need to be stored

    """
    tm = init_miner(miner_config, miner_state)
    df = pd.read_csv(in_csv)
    # df["LineId"] = df.index
    df["Label"] = 0
    df["EventId"] = df.apply(lambda row: inference(row._value, tm), axis=1)
    df["datetime"] = pd.to_datetime(df['_time'])
    df['timestamp'] = df["datetime"].values.astype(np.int64) // 10 ** 9
    
    df = df.sort_values(by=['timestamp']) # sort by timestamp

    df['deltaT'] = df['datetime'].diff() / np.timedelta64(1, 's')
    df['deltaT'].fillna(0)

    df = df[["timestamp", "Label", "EventId", "deltaT"]]

    df.to_csv(os.path.join(out_csv), index=False)

def process(in_csv, outdir, seq_len=20, step_size=1, train_ratio=0.4):
    df = pd.read_csv(in_csv)
    # df = sliding_window(df, para={"window_size": seq_len, "step_size": step_size})

    events = df["EventId"].tolist()
    labels = df["Label"].tolist()

    sequences = []
    for i in range(len(events) - seq_len + 1):
        window = events[i:i + seq_len]
        sequences.append((window, max(labels[i:i + seq_len]))) #sequence plus label

    encoded_sequences = [seq for seq,_ in sequences]
    encoded_sequences = [" ".join(seq) for seq in encoded_sequences]

    normal_sequences = [encoded_sequences[i] for i in range(len(encoded_sequences)) if sequences[i][1] == 0] 
    abnormal_sequences = [encoded_sequences[i] for i in range(len(encoded_sequences)) if sequences[i][1] == 1]
    normal_len = len(normal_sequences)

    shuffle(normal_sequences)
    shuffle(abnormal_sequences)

    train_len = int(normal_len * train_ratio)

    train_sequence = normal_sequences[:train_len]
    test_sequence = normal_sequences[train_len:]

    with open(os.path.join(outdir, "train"), "w") as f:
      for seq in train_sequence:
        f.write(seq + "\n")

    with open(os.path.join(outdir, "test_normal"), "w") as f:
      for seq in test_sequence:
        f.write(seq + "\n")
    
    with open(os.path.join(outdir, "test_abnormal"), "w") as f:
      for seq in abnormal_sequences:
        f.write(seq + "\n")

    # with open(os.path.join(outdir, "word_vocab.json"), "w") as f:
    #     json.dump(event2idx, f)

    # #########
    # # Train #
    # #########
    # df_normal =df[df["Label"] == 0]
    # df_normal = df_normal.sample(frac=1, random_state=12).reset_index(drop=True) #shuffle
    # normal_len = len(df_normal)
    # train_len = int(normal_len * train_ratio)

    # train = df_normal[:train_len]
    # file_generator(os.path.join(outdir,'train'), train, ["EventId"])

    # print("training size {}".format(train_len))


    # ###############
    # # Test Normal #
    # ###############
    # test_normal = df_normal[train_len:]
    # file_generator(os.path.join(outdir, 'test_normal'), test_normal, ["EventId"])
    # print("test normal size {}".format(normal_len - train_len))

    # del df_normal
    # del train
    # del test_normal
    # gc.collect()

    # #################
    # # Test Abnormal #
    # #################
    # df_abnormal = df[df["Label"] == 1]
    # file_generator(os.path.join(outdir,'test_abnormal'), df_abnormal, ["EventId"])
    # print('test abnormal size {}'.format(len(df_abnormal)))
