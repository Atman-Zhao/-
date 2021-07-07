import numpy as np
import wfdb
import matplotlib.pyplot as plt
from scipy.signal import resample
import glob
import pywt

# print(file)
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

#---心拍截取
def heartbeat(file0):
    # for i in range(10):
    N_Seg = SVEB_Seg = VEB_Seg = F_Seg = Q_Seg = []
    # --------去掉指定的四个导联的头文件---------
    De_file = [panth[:-1] + '\\102.hea', panth[:-1] + '\\104.hea', panth[:-1] + '\\107.hea', panth[:-1] + '\\217.hea']
    file = list(set(file0).difference(set(De_file)))
    print(len(file))
    for f in range(len(file)):  #len(file):44
        # print(file[i][-7:-4])
        annotation = wfdb.rdann(panth + file[f][-7:-4], 'atr')
        record_name = annotation.record_name  # 读取记录名称
        # qwe = wfdb.rdsamp(panth + record_name)
        Record_1 = wfdb.rdsamp(panth + record_name)[0][:, 0]  # 一般只取一个导联    360的采样率
        # Record_2 = wfdb.rdsamp(panth + record_name)[0][:, 1]
        record = WTfilt_1d(Record_1)  # 小波去噪    [-1:-3]

        record_1 = record - np.min(record)
        record_2 = record_1 / np.max(record_1)  # 归一化

        label = annotation.symbol  # 心拍标签列表
        label_index = annotation.sample  # 标签索引列表
        # print(label)
        # print(label_index)
        print(len(label_index))
        for j in range(len(label_index)):  #
            if label_index[j] >= 144 and (label_index[j] + 180) <= 650000:
                if label[j] == 'N' or label[j] == '.' or label[j] == 'L' or label[j] == 'R' or label[j] == 'e' or label[
                    j] == 'j':
                    Seg = record_2[label_index[j] - 144:label_index[j] + 180]  # R峰的前0.4s和后0.5s
                    # segment = resample(Seg, 251, axis=0)  # 重采样到251
                    N_Seg.append(Seg)
                if label[j] == 'A' or label[j] == 'a' or label[j] == 'J' or label[j] == 'S':
                    Seg = record_2[label_index[j] - 144:label_index[j] + 180]
                    # segment = resample(Seg, 251, axis=0)
                    SVEB_Seg.append(Seg)
                if label[j] == 'V' or label[j] == 'E':
                    Seg = record_2[label_index[j] - 144:label_index[j] + 180]
                    # segment = resample(Seg, 251, axis=0)
                    VEB_Seg.append(Seg)
                if label[j] == 'F':
                    Seg = record_2[label_index[j] - 144:label_index[j] + 180]
                    # segment = resample(Seg, 251, axis=0)
                    F_Seg.append(Seg)
                if label[j] == '/' or label[j] == 'f' or label[j] == 'Q':
                    Seg = record_2[label_index[j] - 144:label_index[j] + 180]       #324
                    # segment = resample(Seg, 251, axis=0)
                    Q_Seg.append(Seg)

    N_segement = np.array(N_Seg)
    SVEB_segement = np.array(SVEB_Seg)
    VEB_segement = np.array(VEB_Seg)
    F_segement = np.array(F_Seg)
    # Q_segement = np.array(Q_Seg)

    label_N = np.zeros(N_segement.shape[0], dtype=int)
    label_SVEB = np.ones(SVEB_segement.shape[0], dtype=int)
    label_VEB = np.ones(VEB_segement.shape[0], dtype=int) * 2
    label_F = np.ones(F_segement.shape[0], dtype=int) * 3
    # label_Q = np.ones(Q_segement.shape[0], dtype=int) * 4

    Data = np.concatenate((N_segement, SVEB_segement, VEB_segement, F_segement), axis=0)
    Label = np.concatenate((label_N, label_SVEB, label_VEB, label_F), axis=0)

    return Data, Label

panth = 'D:/Users/zhao_/Desktop/graduation project/ECG_Classification/mit-bih-arrhythmia-database-1.0.0/'
file = glob.glob(panth + '*.hea')
Data, Label = heartbeat(file)

Data = np.save('D:/Users/zhao_/Desktop/graduation project/ECG_Classification/' + 'Data', Data)
Label = np.save('D:/Users/zhao_/Desktop/graduation project/ECG_Classification/' + 'Label', Label)
