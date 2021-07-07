#AA AA 12 02 C8 03 3B 84 05 00 F9 00 03 44 08 33 85 03 FF FF E5 88
import serial
import matplotlib.pyplot as plt
import time
data = [0xAA, 0xAA, 0x12, 0x02, 0xC8, 0x03, 0x3B, 0x84, 0x05, 0x00, 0xF9, 0x00, 0x03,
        0x44, 0x08, 0x33, 0x85, 0x03, 0xFF, 0xFF, 0xE5, 0x88]
def parsePayload( payload,  pLength):
    #循环，直到从payload[]中解析所有字节…
    bytesParsed = 0
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

        print(f"EXCODE level: {extendedCodeLevel} CODE: {code} length: {length}\n")
        print( "Data value(s):" )
        for i in range(length):
            print(f"{payload[bytesParsed+i] & 0xFF}\n")
        bytesParsed += length
    return 0


# portx = 'COM5'
# bps = 57600
# timex = None
# ser = serial.Serial(portx, bps, timeout=timex)
# print(f'串口详情：{ser}\n')
# time.sleep(1)

SYNC = 0xAA
while True:
    if data[0] == SYNC:  #不断读取新的字节，知道读到[SYNC]时才执行下一步。
        if data[1] == SYNC:  #读取下一个字节的值，确保其为[SYNC]
            pLength = data[2]    #读取下一个字节，将其视作[PLENGTH]
            while True:
                 if pLength != 170:  #如果[PLENGTH]是170（[SYNC]）,重复第
                     break
            if pLength < 169:   #如果[PLENGTH]比170 大，返回第一步（PLENGTH 太大）
                payload = [data[i+3] for i in range(pLength)] #读取[PLENGTH]后面的一个字节（其属于有效数据），把其保存到存储空间
                checksum = 0
                for i in range(pLength):    #把其全部加起来（checksum += byte）。
                    checksum += payload[i]
                checksum &= 0xFF
                checksum = ~checksum & 0xFF
                if data[2+pLength+1] == checksum:
                    parsePayload(payload, pLength)
# plt.plot(a)
#print(a)

#ser.close()
    print('关闭串口')

