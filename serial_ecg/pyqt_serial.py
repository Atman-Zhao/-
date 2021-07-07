from PyQt5 import QtWidgets,QtCore,QtGui
import pyqtgraph as pg
import sys
import traceback
import psutil
import serial
import matplotlib.pyplot as plt
import time

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
        self.plot_plt.setYRange(min = -10000, max = 10000)
        self.data_list = []

    def ReadDataPackge(self, ser):
        SYNC = 0xAA
        while True:
            if ser.read()[0] == SYNC:  # 不断读取新的字节，知道读到[SYNC]时才执行下一步。
                if ser.read()[0] == SYNC:  # 读取下一个字节的值，确保其为[SYNC]
                    pLength = ser.read()[0]  # 读取下一个字节，将其视作[PLENGTH]
                    while True:
                        if pLength != 170:  # 如果[PLENGTH]是170（[SYNC]）,重复第
                            break
                    if pLength < 169:  # 如果[PLENGTH]比170 大，返回第一步（PLENGTH 太大）
                        payload = [ser.read()[0] for i in range(pLength)]  # 读取[PLENGTH]后面的一个字节（其属于有效数据），把其保存到存储空间
                        checksum = 0
                        for i in range(pLength):  # 把其全部加起来（checksum += byte）。
                            checksum += payload[i]
                        checksum &= 0xFF
                        checksum = ~checksum & 0xFF
                        if ser.read()[0] == checksum:


                            bytesParsed = 0
                            while bytesParsed < pLength:
                                # 解析代码和长度
                                extendedCodeLevel = 0
                                EXCODE = 0x55
                                while payload[bytesParsed] == EXCODE:  # 提取数据单元开头中，[EXCODE]值的个数。
                                    extendedCodeLevel += 1
                                    bytesParsed += 1

                                code = payload[bytesParsed]  # 提取当前行数据中[CODE]的值。
                                bytesParsed += 1

                                if (code & 0x80):  # 如果[CODE] >= 0x80，把下一个字节当成[VLENGTH]来解释。
                                    length = payload[bytesParsed]
                                    bytesParsed += 1
                                else:
                                    length = 1

                                if code == 0x10:
                                    continue
                                # print(f"EXCODE level: {extendedCodeLevel} CODE: {code} length: {length}\n")
                                # print( "Data value(s):" )
                                value = [0, 0]
                                for i in range(length):
                                    if (code == 0x80):
                                        # print(f"数值:{payload[bytesParsed+i] & 0xFF}\n")
                                        value[i] = payload[bytesParsed + i]

                                bytesParsed += length

                                raw = value[0] * 256 + value[1]
                                if (raw >= 32768):
                                    raw = raw - 65536

                                self.data_list.append(raw)
                                # y.append(raw)

                                self.plot_plt.plot().setData(self.data_list, pen='g')
                                time.sleep(0.01)
                                print(len(self.data_list))


def main():
    try:
        app = QtWidgets.QApplication(sys.argv)
        gui = MainUi()
        ser = serial.Serial('COM5', 57600, timeout=None)
        print(f'串口详情：{ser}\n')
        gui.show()
        gui.ReadDataPackge(ser)
        sys.exit(app.exec_())

    except Exception as e:
        print(f"异常：{e}")


if __name__ == '__main__':
    while True:
        main()