import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import serial
import sys
import pywt
import heartpy as hp

def WTfilt_1d(sig):
    """
    对信号进行小波变换滤波
    :param sig: 输入信号，1-d array
    :return: 小波滤波后的信号，1-d array
    """
    coeffs = pywt.wavedec(sig, 'db6', level=6)
    coeffs[-1] = np.zeros(len(coeffs[-1]))
    coeffs[-2] = np.zeros(len(coeffs[-2]))
    coeffs[0] = np.zeros(len(coeffs[0]))
    coeffs[1] = np.zeros(len(coeffs[1]))
    sig_filt = pywt.waverec(coeffs, 'db6')
    return sig_filt

def GetData(ser):
    SYNC = 0xAA
    while True:
        if ser.read()[0] == SYNC:  # 不断读取新的字节，知道读到[SYNC]时才执行下一步。
            if ser.read()[0] == SYNC:  # 读取下一个字节的值，确保其为[SYNC]
                pLength = ser.read()[0]  # 读取下一个字节，将其视作[PLENGTH]

                payload = [ser.read()[0] for i in range(pLength)]  # 读取[PLENGTH]后面的一个字节（其属于有效数据），把其保存到存储空间
                checksum = 0
                for i in range(pLength):  # 把其全部加起来（checksum += byte）。
                    checksum += payload[i]
                checksum &= 0xFF
                checksum = ~checksum & 0xFF
                if ser.read()[0] == checksum:

                    code = payload[0]  # 提取当前行数据中[CODE]的值。
                    length = payload[1]
                    value = [0, 0]
                    if (code == 0x80):
                        for i in range(length):
                            # print(f"数值:{payload[bytesParsed+i] & 0xFF}\n")
                            value[i] = payload[2+i]

                    raw = value[0] * 256 + value[1]
                    if (raw >= 32768):
                        raw = raw - 65536
                    return  raw


def update1():
    global data1, ptr1, ser, i, serial_data, output_data, flag, length

    if flag == True:
        data1[:-1] = data1[1:]
        data1[-1] = output_data[i]
        curve1.setData(data1)

    serial_data.append(GetData(ser))
    i += 1
    if i == length:
        # 带通滤波 [0.75, 15]
        output_data = hp.filter_signal(serial_data, cutoff=[8, 120], sample_rate=512 , order=3, filtertype='bandpass')
        #output_data = WTfilt_1d(serial_data)
        serial_data = []
        flag = True
        i = 0



## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    while True:
        try:
            length = 1500
            i = flag = 0
            serial_data=output_data=[]
            ser = serial.Serial('COM5', 57600, timeout=0.5)
            print(f'串口详情：{ser}\n')
            win = pg.GraphicsLayoutWidget(show=True)
            win.setWindowTitle('Scrolling Plots Mode 1')
            p1 = win.addPlot()
            data1 = np.zeros(length)

            p1.setYRange(-10000, 10000)
            curve1 = p1.plot(data1)

            timer = pg.QtCore.QTimer()
            timer.timeout.connect(update1)
            timer.start(0)

            if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
                QtGui.QApplication.instance().exec_()
                ser.close()
                print('关闭串口')
                break

        except Exception as e:
            #global ser
            print(f"异常：{e}")
            #ser.close()
            print('关闭串口')

