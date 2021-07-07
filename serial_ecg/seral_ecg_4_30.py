#AA AA 12 02 C8 03 3B 84 05 00 F9 00 03 44 08 33 85 03 FF FF E5 88
import serial
import matplotlib.pyplot as plt
import time

from PyQt5 import QtWidgets,QtCore,QtGui
import pyqtgraph as pg
import sys
import traceback
import psutil
from queue import Queue

class MainUi(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CPU使用率监控 - 州的先生https://zmister.com")
        self.main_widget = QtWidgets.QWidget()  # 创建一个主部件
        self.main_layout = QtWidgets.QGridLayout()  # 创建一个网格布局
        self.main_widget.setLayout(self.main_layout)  # 设置主部件的布局为网格
        self.setCentralWidget(self.main_widget)  # 设置窗口默认部件

        self.plot_widget = QtWidgets.QWidget()  # 实例化一个widget部件作为K线图部件
        self.plot_layout = QtWidgets.QGridLayout()  # 实例化一个网格布局层
        self.plot_widget.setLayout(self.plot_layout)  # 设置K线图部件的布局层
        self.plot_plt = pg.PlotWidget()  # 实例化一个绘图部件
        self.plot_plt.showGrid(x=True,y=True) # 显示图形网格
        self.plot_layout.addWidget(self.plot_plt)  # 添加绘图部件到K线图部件的网格布局层
        # 将上述部件添加到布局层中
        self.main_layout.addWidget(self.plot_widget, 1, 0, 3, 3)

        self.setCentralWidget(self.main_widget)
        self.plot_plt.setYRange(max=100,min=0)
        self.data_list = []
        self.timer_start()

x = []
y = []
j = 0

def parsePayload( payload,  pLength):
    #循环，直到从payload[]中解析所有字节…
    global x
    global y
    global j
    bytesParsed = 0
    plt.ion()

    #print('123')
    while bytesParsed < pLength:
        # 解析代码和长度
        extendedCodeLevel = 0
        EXCODE = 0x55
        while payload[bytesParsed] == EXCODE:   #提取数据单元开头中，[EXCODE]值的个数。
            extendedCodeLevel += 1
            bytesParsed += 1

        code = payload[bytesParsed]   #提取当前行数据中[CODE]的值。
        bytesParsed += 1

        if( code & 0x80 ):  #如果[CODE] >= 0x80，把下一个字节当成[VLENGTH]来解释。
            length = payload[bytesParsed]
            bytesParsed += 1
        else:
            length = 1

        if code == 0x10:
            continue
        #print(f"EXCODE level: {extendedCodeLevel} CODE: {code} length: {length}\n")
        #print( "Data value(s):" )
        value = [0, 0]
        for i in range(length):
            if (code == 0x80):
                #print(f"数值:{payload[bytesParsed+i] & 0xFF}\n")
                value[i] = payload[bytesParsed + i]
            # if (code == 0x02):
            #     #print(f"噪声:{payload[bytesParsed+i] & 0xFF}\n")
            #     POOR_SIGNAL = payload[bytesParsed+i]
            # if (code == 0x03):
            #     #print(f"心率:{payload[bytesParsed+i] & 0xFF}\n")
            #     HEART_RATE = payload[bytesParsed+i]
            # if (code == 0x08):
            #     #print(f"[CONFIG_BYTE]:{payload[bytesParsed+i] & 0xFF}\n")
            #     CONFIG_BYTE = payload[bytesParsed+i]
            # if (code == 0x84):
            #     #print(f"[DEBUG_1]:{payload[bytesParsed+i] & 0xFF}\n")
            #     DEBUG_1 = payload[bytesParsed + i]
            # if (code == 0x85):
            #     #print(f"[DEBUG_2]:{payload[bytesParsed+i] & 0xFF}\n")
            #     DEBUG_2 = payload[bytesParsed + i]

        bytesParsed += length

        raw = value[0] * 256 + value[1]
        if (raw >= 32768):
            raw = raw - 65536

        x.append(j)
        j = j + 1
        y.append(raw)


        # print(len(y))
        # if len(y) % 1000 == 0:
        #     # print(y)
        #     plt.clf()
        #     plt.xlim(0, 2000)
        #     plt.ylim(-15000, 15000)
        #     plt.plot(x, y)
        #     plt.pause(0.1)
        #     plt.ioff()     #关闭交互模式
        #     if len(y) == 2000:
        #         j = 0
        #         x = []
        #         y = []
    return 0

def ReadDataPackge(ser):
    SYNC = 0xAA
    # for i in range(10):
    #     print(ser.read().encode())
    #     # if ser.read() == 0xaa:
    #     #     print('yes')

    while True:
        if ser.read()[0] == SYNC:  #不断读取新的字节，知道读到[SYNC]时才执行下一步。
            if ser.read()[0]  == SYNC:  #读取下一个字节的值，确保其为[SYNC]
                pLength = ser.read()[0]     #读取下一个字节，将其视作[PLENGTH]
                while True:
                     if pLength != 170:  #如果[PLENGTH]是170（[SYNC]）,重复第
                         break
                if pLength < 169:   #如果[PLENGTH]比170 大，返回第一步（PLENGTH 太大）
                    payload = [ser.read()[0]  for i in range(pLength)] #读取[PLENGTH]后面的一个字节（其属于有效数据），把其保存到存储空间
                    checksum = 0
                    for i in range(pLength):    #把其全部加起来（checksum += byte）。
                        checksum += payload[i]
                    checksum &= 0xFF
                    checksum = ~checksum & 0xFF
                    if ser.read()[0]  == checksum:
                        parsePayload(payload, pLength)

    # plt.plot(a)
    #print(a)

def main():
    try:
        app = pg.mkQApp()  # 建立app
        win = pg.GraphicsWindow()  # 建立窗口
        win.setWindowTitle(u'pyqtgraph逐点画波形图')
        win.resize(800, 500)  # 小窗口大小
        data = array.array('i')  # 可动态改变数组的大小,double型数组
        historyLength = 100  # 横坐标长度
        a = 0
        data = np.zeros(historyLength).__array__('d')  # 把数组长度定下来
        p = win.addPlot()  # 把图p加入到窗口中
        p.showGrid(x=True, y=True)  # 把X和Y的表格打开
        p.setRange(xRange=[0, historyLength], yRange=[-50, 50], padding=0)
        p.setLabel(axis='left', text='y / V')  # 靠左
        p.setLabel(axis='bottom', text='x / point')
        p.setTitle('semg')  # 表格的名字
        curve = p.plot()  # 绘制一个图形
        curve.setData(data)

        #mSerial.flushInput()  # 清空缓冲区

        ser = serial.Serial('COM5', 57600, timeout=None)
        print(f'串口详情：{ser}\n')

        th1 = threading.Thread(target=Serial)
        th1.start()



        ReadDataPackge(ser)
        ser.close()
        print('关闭串口')

    except Exception as e:
        #ser.close()
        #print('关闭串口')
        print(f"异常：{e}")

while True:
    main()




