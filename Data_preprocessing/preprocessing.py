import numpy as np
import wfdb
import matplotlib.pyplot as plt
from scipy.signal import resample
import glob
import pywt

# --------------------小波去噪-----------------
def WTfilt_1d(sig):
    """
    对信号进行小波变换滤波
    :param sig: 输入信号，1-d array
    :return: 小波滤波后的信号，1-d array
    """
    coeffs = pywt.wavedec(sig, 'db6', level=9)
    coeffs[-1] = np.zeros(len(coeffs[-1]))
    coeffs[-2] = np.zeros(len(coeffs[-2]))
    coeffs[0] = np.zeros(len(coeffs[0]))
    sig_filt = pywt.waverec(coeffs, 'db6')
    return sig_filt

def hearbeat_classification(file):
    Normal = [];Non_ectopic = [];Supraventriculr = [];Ventricular = [];Fusion = []
    for f in range(len(file)):  #
        annotation = wfdb.rdann(panth + file[f][-7:-4], 'atr')
        record_name = annotation.record_name  # 读取记录名称
        Record = wfdb.rdsamp(panth + record_name)[0][:, 0]  # 一般只取一个导联

        record_1 = WTfilt_1d(Record)  # 小波去噪
        record_2 = record_1 - np.min(record_1)
        record = record_2 / np.max(record_2)  # 归一化

        label = annotation.symbol  # 心拍标签列表
        label_index = annotation.sample  # 标签索引列表
        for j in range(len(label_index)):
            if label_index[j] >= 144 and (label_index[j] + 144) <= 650000:      #288
                Seg = record[label_index[j] - 144:label_index[j] + 144]  # R峰的前0.4s和后0.5s
                # segment = resample(Seg, 251, axis=0)  # 重采样到251
                if label[j] == 'N'  :
                    Normal.append(Seg)              #正常
                if label[j] == '.' or label[j] == 'L' or label[j] == 'R' or label[j] == 'e' or label[j] == 'j':
                    Non_ectopic.append(Seg)         #非异位
                if label[j] == 'A' or label[j] == 'a' or label[j] == 'J' or label[j] == 'S':
                    Supraventriculr.append(Seg)     #室上直肠搏动
                if label[j] == 'V' or label[j] == 'E':
                    Ventricular.append(Seg)         #心室异位搏动
                if label[j] == 'F':
                    Fusion.append(Seg)              #融合拍
                # if label[j] == '/' or label[j] == 'f' or label[j] == 'Q':
                #     Unknown.append(Seg)             #未知

    Normal_seg = np.array(Normal)
    Non_ectopic_seg = np.array(Non_ectopic)
    Supraventriculr_seg = np.array(Supraventriculr)
    Ventricular_seg = np.array(Ventricular)
    Fusion_seg = np.array(Fusion)

    Normal_label = np.zeros(Normal_seg.shape[0], dtype=int)
    Non_ectopic_label = np.ones(Non_ectopic_seg.shape[0], dtype=int)
    Supraventriculr_label = np.ones(Supraventriculr_seg.shape[0], dtype=int) * 2
    Ventricular_label = np.ones(Ventricular_seg.shape[0], dtype=int) * 3
    Fusion_label = np.ones(Fusion_seg.shape[0], dtype=int) * 4

    Data = np.concatenate((Normal_seg, Non_ectopic_seg, Supraventriculr_seg, Ventricular_seg, Fusion_seg), axis=0)
    Label = np.concatenate((Normal_label, Non_ectopic_label, Supraventriculr_label, Ventricular_label,Fusion_label), axis=0)

    return Data, Label

def heartbeat(file0):
    '''
    file0:下载的MITAB数据
    '''
    # --------去掉指定的四个导联的头文件---------
    De_file = [panth[:-1] + '\\102.hea', panth[:-1] + '\\104.hea', panth[:-1] + '\\107.hea', panth[:-1] + '\\217.hea']
    file = list(set(file0).difference(set(De_file)))

    train_size = int(0.8*len(file))     #8:2数据集
    train_file = [file[i] for i in range(train_size)]       #训练集
    test_file = list(set(file).difference(set(train_file)))     #测试集
    print(len(train_file))
    print(len(test_file))

    train_data, train_label = hearbeat_classification(train_file)
    test_data, test_label = hearbeat_classification(test_file)

    return train_data, test_data, train_label, test_label

# -----------------------心拍截取和保存---------------------
# 建议一次性截取和保存，不需要重复操作，下次训练和测试的时候，直接load
panth = 'D:/Users/zhao_/Desktop/graduation project/ECG_Classification/mit-bih-arrhythmia-database-1.0.0/'
file = glob.glob(panth + '*.hea')
train_data, test_data, train_label, test_label = heartbeat(file)

train_data = np.save('D:/Users/zhao_/Desktop/graduation project/ECG_Classification/Data_preprocessing/' + 'train_data', train_data)
test_data = np.save('D:/Users/zhao_/Desktop/graduation project/ECG_Classification/Data_preprocessing/' + 'test_data', test_data)
train_label = np.save('D:/Users/zhao_/Desktop/graduation project/ECG_Classification/Data_preprocessing/' + 'train_label', train_label)
test_label = np.save('D:/Users/zhao_/Desktop/graduation project/ECG_Classification/Data_preprocessing/' + 'test_label', test_label)


