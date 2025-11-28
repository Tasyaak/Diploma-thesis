from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import math
import time
import serial
from serial.tools import list_ports
import matplotlib.pyplot as plt

class dataReceive(QObject):
    newCoordinate = pyqtSignal(object)

    def __init__(self, allow_receive, active_COM, start_cmd):
        super().__init__()
        self.allow_receive = allow_receive
        self.active_COM = active_COM
        self.start_cmd = start_cmd

    def getcfgData(self, allow_receive, active_COM, start_cmd):
        self.allow_receive = allow_receive
        self.active_COM = active_COM
        self.start_cmd = start_cmd
    
    def Track(self):
        print(self.allow_receive)
        print(self.active_COM)
        print(self.start_cmd)
        
        while True:
            if (self.allow_receive):
                try:
                    with serial.Serial(self.active_COM, 38400, timeout=5) as ser:
                        ser.write(self.start_cmd.encode('utf-8'))
                        #self.textEdit_2.append(self.start_cmd)
                        s = ser.read(256000)
                    print('receive')
                    print(len(s))
                    self.newCoordinate.emit(s)
                except Exception as e:
                    self.allow_receive = False
                    print(e)
            QThread.msleep(10)


class Ui_MainWindow(QWidget):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(820, 676)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")

        self.newAction = QAction(self)
        self.menuBar = QMenuBar(self)
        self.menuBar.setGeometry(QRect(0, 0, 820, 20))
        self.shadow = QGraphicsDropShadowEffect(self,
            blurRadius = 7.5,
            color = QColor(40, 40, 40),
            offset = QPointF(0.5, 0.5)
        )
        self.menuBar.setGraphicsEffect(self.shadow)
        self.openAction = QAction("Открыть", self)
        self.exitAction = QAction("Выход", self)
        self.fileMenu = QMenu('Файл', self)
        self.menuBar.addMenu(self.fileMenu)
        self.fileMenu.addAction(self.openAction)
        self.fileMenu.addAction(self.exitAction)

        self.openAction.triggered.connect(self.showDialog)
        self.exitAction.triggered.connect(self.close)
        
        self.startButton = QPushButton(self.centralwidget)
        self.startButton.setObjectName(u"startButton")
        self.startButton.setGeometry(QRect(280, 620, 75, 23))
        self.stopButton = QPushButton(self.centralwidget)
        self.stopButton.setObjectName(u"stopButton")
        self.stopButton.setGeometry(QRect(380, 620, 75, 23))
        self.lineEdit_zeroed_periods = QLineEdit(self.centralwidget)
        self.lineEdit_zeroed_periods.setObjectName(u"lineEdit_zeroed_periods")
        self.lineEdit_zeroed_periods.setGeometry(QRect(700, 110, 71, 20))
        self.lineEdit_number_of_periods = QLineEdit(self.centralwidget)
        self.lineEdit_number_of_periods.setObjectName(u"lineEdit_number_of_periods")
        self.lineEdit_number_of_periods.setGeometry(QRect(700, 160, 71, 20))
        self.waveform_comboBox = QComboBox(self.centralwidget)
        self.waveform_comboBox.setObjectName(u"waveform_comboBox")
        self.waveform_comboBox.setGeometry(QRect(700, 50, 69, 22))
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(700, 25, 81, 20))
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(700, 88, 61, 16))
        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(700, 136, 61, 20))
        self.lineEdit_dac_file_path = QLineEdit(self.centralwidget)
        self.lineEdit_dac_file_path.setObjectName(u"lineEdit_dac_file_path")
        self.lineEdit_dac_file_path.setGeometry(QRect(50, 220, 351, 20))
        self.label_4 = QLabel(self.centralwidget)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(700, 190, 71, 16))
        self.lineEdit_duty = QLineEdit(self.centralwidget)
        self.lineEdit_duty.setObjectName(u"lineEdit_duty")
        self.lineEdit_duty.setGeometry(QRect(700, 210, 71, 20))
        self.pushButton_dacLoad = QPushButton(self.centralwidget)
        self.pushButton_dacLoad.setObjectName(u"pushButton_dacLoad")
        self.pushButton_dacLoad.setGeometry(QRect(507, 220, 75, 20))
        self.pushButton_dac_path_load = QPushButton(self.centralwidget)
        self.pushButton_dac_path_load.setObjectName(u"pushButton_dac_path_load")
        self.pushButton_dac_path_load.setGeometry(QRect(410, 220, 75, 20))
        self.pushButton_dac_path_load.clicked.connect(self.showDialog)
        self.label_5 = QLabel(self.centralwidget)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(700, 265, 47, 13))
        self.label_6 = QLabel(self.centralwidget)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(700, 280, 91, 16))
        self.label_7 = QLabel(self.centralwidget)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setGeometry(QRect(700, 307, 47, 13))
        self.lineEdit_adc_freq = QLineEdit(self.centralwidget)
        self.lineEdit_adc_freq.setObjectName(u"lineEdit_adc_freq")
        self.lineEdit_adc_freq.setGeometry(QRect(700, 330, 71, 20))
        self.label_8 = QLabel(self.centralwidget)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setGeometry(QRect(700, 363, 61, 16))
        self.lineEdit_dac_freq = QLineEdit(self.centralwidget)
        self.lineEdit_dac_freq.setObjectName(u"lineEdit_dac_freq")
        self.lineEdit_dac_freq.setGeometry(QRect(700, 385, 71, 20))
        self.lineEdit_amplitude = QLineEdit(self.centralwidget)
        self.lineEdit_amplitude.setObjectName(u"lineEdit_amplitude")
        self.lineEdit_amplitude.setGeometry(QRect(600, 220, 71, 20))
        self.textEdit = QTextEdit(self.centralwidget)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setGeometry(QRect(400, 470, 261, 101))
        self.textEdit_2 = QTextEdit(self.centralwidget)
        self.textEdit_2.setObjectName(u"textEdit_2")
        self.textEdit_2.setGeometry(QRect(50, 470, 256, 101))
        self.label_9 = QLabel(self.centralwidget)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setGeometry(QRect(400, 447, 50, 15))
        self.label_10 = QLabel(self.centralwidget)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setGeometry(QRect(50, 447, 71, 16))
        self.label_11 = QLabel(self.centralwidget)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setGeometry(QRect(600, 203, 71, 16))

        self.startButton.clicked.connect(self.getDataStart)
        self.pushButton_dacLoad.clicked.connect(self.runGraph)
        self.stopButton.clicked.connect(self.stopMeasure)
        self.waveform_comboBox.addItems(['sin', 'triangle', 'square', 'sawtooth', 'other'])
        
        self.intype_comboBox = QComboBox(self.centralwidget)
        self.intype_comboBox.setObjectName(u"intype_comboBox")
        self.intype_comboBox.setGeometry(QRect(700, 440, 69, 22))
        self.intype_comboBox.addItems(['text', 'uint8_t', 'uint8_t split'])

        self.trig_comboBox = QComboBox(self.centralwidget)
        self.trig_comboBox.setObjectName(u"trig_comboBox")
        self.trig_comboBox.setGeometry(QRect(700, 472, 75, 22))
        self.trig_comboBox.addItems(['w trigger', 'w/o trigger'])

        self.setButton = QPushButton(self.centralwidget)
        self.setButton.setObjectName(u"set")
        self.setButton.setGeometry(QRect(700, 565, 75, 23))
        self.setButton.clicked.connect(self.configFunction)

        self.curCfgButton = QPushButton(self.centralwidget)
        self.curCfgButton.setObjectName(u"set")
        self.curCfgButton.setGeometry(QRect(700, 600, 90, 23))
        self.curCfgButton.clicked.connect(self.showCurrent)

        self.line = QFrame(self.centralwidget)
        self.line.setObjectName(u"line")
        self.line.setGeometry(QRect(700, 250, 71, 3))
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.line_2 = QFrame(self.centralwidget)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setGeometry(QRect(700, 423, 71, 3))
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.line_3 = QFrame(self.centralwidget)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setGeometry(QRect(681, 40, 20, 610))
        self.line_3.setFrameShape(QFrame.VLine)
        self.line_3.setFrameShadow(QFrame.Sunken)

        self.checkBox = QCheckBox(self.centralwidget)
        self.checkBox.setObjectName(u"checkBox")
        self.checkBox.setGeometry(QRect(690, 540, 70, 17))

        self.checkBox_2 = QCheckBox(self.centralwidget)
        self.checkBox_2.setObjectName(u"checkBox_2")
        self.checkBox_2.setGeometry(QRect(570, 585, 120, 17))

        self.checkBox_3 = QCheckBox(self.centralwidget)
        self.checkBox_3.setObjectName(u"checkBox_3")
        self.checkBox_3.setGeometry(QRect(745, 540, 70, 17))

        self.checkBox_4 = QCheckBox(self.centralwidget)
        self.checkBox_4.setObjectName(u"checkBox_4")
        self.checkBox_4.setGeometry(QRect(190, 625, 90, 17))

        self.lineEdit_trigger = QLineEdit(self.centralwidget)
        self.lineEdit_trigger.setObjectName(u"lineEdit_trigger")
        self.lineEdit_trigger.setGeometry(QRect(700, 515, 71, 20))

        self.label_12 = QLabel(self.centralwidget)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setGeometry(QRect(700, 499, 71, 13))

        self.uart_list_comboBox = QComboBox(self)
        self.uart_list_comboBox.setGeometry(QRect(700, 635, 75, 23))
        com_list = list_ports.comports()
        com_names = []

        self.lineEdit_adc_file_path = QLineEdit(self.centralwidget)
        self.lineEdit_adc_file_path.setObjectName(u"lineEdit_adc_file_path")
        self.lineEdit_adc_file_path.setGeometry(QRect(50, 585, 420, 20))

        self.pushButton_clear = QPushButton(self.centralwidget)
        self.pushButton_clear.setObjectName(u"pushButton__clear")
        self.pushButton_clear.setGeometry(QRect(315, 535, 75, 23))
        self.pushButton_clear.clicked.connect(self.clearOut)

        self.pushButton_plot = QPushButton(self.centralwidget)
        self.pushButton_plot.setObjectName(u"pushButton__plot")
        self.pushButton_plot.setGeometry(QRect(315, 490, 75, 23))
        self.pushButton_plot.clicked.connect(self.pyplot)

        self.pushButton_adc_path_load = QPushButton(self.centralwidget)
        self.pushButton_adc_path_load.setObjectName(u"pushButton_adc_path_load")
        self.pushButton_adc_path_load.setGeometry(QRect(485, 585, 75, 20))
        self.pushButton_adc_path_load.clicked.connect(self.showDialog_2)

        self.pushButton_uart = QPushButton(self.centralwidget)
        self.pushButton_uart.setObjectName(u"pushButton_uart")
        self.pushButton_uart.setGeometry(QRect(600, 635, 75, 20))
        self.pushButton_uart.clicked.connect(self.refresh)
        
        for i in range(len(list_ports.comports())):
            com_names.append(list_ports.comports()[i].device)
        self.uart_list_comboBox.addItems(com_names)
        
        self.out_arr = []
        self.lineEdit_number_of_periods.setText('5')
        self.lineEdit_zeroed_periods.setText('2')
        self.lineEdit_duty.setText('0.5')
        self.lineEdit_adc_freq.setText('10000000')
        self.lineEdit_dac_freq.setText('1000000')
        self.lineEdit_trigger.setText('10')
        self.lineEdit_amplitude.setText('0.2')
        
        self.receive = False
        
        self.thread = QThread()
        self.dataReceive = dataReceive(self.receive, f'{self.uart_list_comboBox.currentText()}', 'start')
        self.dataReceive.moveToThread(self.thread)
        self.dataReceive.newCoordinate.connect(self.printDebug)
        self.thread.started.connect(self.dataReceive.Track)
        self.thread.start()

        self.adc_filename = ''
        self.receive_arr = []
        self.read_config = ''
        self.drawflag = False
        
        
        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)
    
    @pyqtSlot(object)
    def printDebug(self, string):
        try:
            self.textEdit.append(str(string))
            if (self.intype_comboBox.currentText() == 'text'):
                in_arr = str(string)
                in_arr = in_arr.replace('\\x00', '')
                in_arr = in_arr.replace("b'", '')
                in_arr = in_arr.replace("'", '')
                in_arr = in_arr[:-2].split('\\n')
                for i in range(len(in_arr)):
                    in_arr[i] = ((int(in_arr[i])/4095) - 0.5)
                self.receive_arr = in_arr
                print(len(in_arr))
            elif (self.intype_comboBox.currentText() == 'uint8_t'):
                in_arr = []
                for i in range(len(string)):
                    if (string[i] != 0):
                        in_arr.append((string[i]/255) - 0.5)
                print(len(in_arr))
                self.receive_arr = in_arr

            elif (self.intype_comboBox.currentText() == 'uint8_t split'):
                in_arr = []
                for i in range(len(string)//2):
                    val = (string[i*2] << 8 )| string[i*2+1]
                    if (val != 0):
                        in_arr.append((val/4095) - 0.5)
                print(len(in_arr))
                self.receive_arr = in_arr
                    
            if bool(self.checkBox_2.checkState()):
                str_to_file = f'\n{time.ctime()}\n{self.read_config}\n#Start\n'
                for i in range(len(self.receive_arr)):
                    str_to_file += f'{self.receive_arr[i]}\n'
                print(self.adc_filename)
                with open(f'{self.lineEdit_adc_file_path.text()}/{self.adc_filename.replace(":", ".")}.txt', 'a+') as f:
                    f.write(str_to_file)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка ", f"{e}", QMessageBox.Ok)
        

    def refresh(self):
        print('refresh')
        self.com_names = []
        for i in range(len(list_ports.comports())):
            self.com_names.append(list_ports.comports()[i].device)
        self.uart_list_comboBox.clear()
        self.uart_list_comboBox.addItems(self.com_names)
        
    def stopMeasure(self):
        self.receive = False
        self.dataReceive.getcfgData(self.receive, f'{self.uart_list_comboBox.currentText()}', 'start')

    def pyplot(self):
        try:
            if len(self.receive_arr) != 0:
                fig = plt.figure()
                fig1 = fig.add_subplot(221)
                pl = []
                for i in range(len(self.receive_arr)):
                    pl.append(-1*self.receive_arr[i])
                x = range(len(pl))
                fig1.plot(x,pl)
                plt.show()
            else:
                QMessageBox.critical(self, "Ошибка ", 'График пуст', QMessageBox.Ok)
        except Exception as e:
            print(e)

    def showCurrent(self):
        try:
            with serial.Serial(f'{self.uart_list_comboBox.currentText()}', 38400, timeout=1) as ser:
                ser.write('current_config'.encode('utf-8'))
                s = ser.read(1000)
            self.textEdit_2.append('current_config')
            s = str(s)
            s = s[:s.rfind('\\n')]
            s = s.replace("b'", '')
            s = s.replace('\\x00', '')
            s = s.replace('\\n', '\n')
            self.textEdit.append(s)
            return s
        except Exception as e:
            QMessageBox.critical(self, "Ошибка ", f"{e}", QMessageBox.Ok)

    def configFunction(self):
        try:
            adc_freq = self.lineEdit_adc_freq.text()
            dac_freq = self.lineEdit_dac_freq.text()
            out_type = 0
            trigger_val = self.lineEdit_trigger.text()
            relay_state = int(bool(self.checkBox.checkState()))
            dac_dma_trig = int(bool(self.checkBox_3.checkState()))
            if (len(adc_freq) < 8):
                zero_str = '0'
                zero_str *= 8 - len(adc_freq) 
                adc_freq = zero_str + adc_freq

            if (len(dac_freq) < 8):
                zero_str = '0'
                zero_str *= 8 - len(dac_freq) 
                dac_freq = zero_str + dac_freq

            if (len(trigger_val) < 3):
                zero_str = '0'
                zero_str *= 3 - len(trigger_val) 
                trigger_val = zero_str + trigger_val    

            if (self.intype_comboBox.currentText() == 'text'):
                if (self.trig_comboBox.currentText() == 'w trigger'):
                    out_type = 2
                elif (self.trig_comboBox.currentText() == 'w/o trigger'):
                    out_type = 0
            elif (self.intype_comboBox.currentText() == 'uint8_t'):
                if (self.trig_comboBox.currentText() == 'w trigger'):
                    out_type = 3
                elif (self.trig_comboBox.currentText() == 'w/o trigger'):
                    out_type = 1

            elif (self.intype_comboBox.currentText() == 'uint8_t split'):
                if (self.trig_comboBox.currentText() == 'w trigger'):
                    out_type = 5
                elif (self.trig_comboBox.currentText() == 'w/o trigger'):
                    out_type = 4
            
            out_str = f'set{adc_freq}_{dac_freq}_{out_type}_0_{trigger_val}_{relay_state}_{dac_dma_trig}'
            with serial.Serial(f'{self.uart_list_comboBox.currentText()}', 38400, timeout=2) as ser:
                ser.write(out_str.encode('utf-8'))
                s = ser.read(1000)
            s = str(s)
            s = s.replace("b'", '')
            s = s.replace('\\x00', '')
            s = s.replace('\\n', '\n')
            s = s.replace("'", '')
            self.textEdit_2.append(out_str)
            self.textEdit.append(s)
            
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка ", f"{e}", QMessageBox.Ok)
        
                
            
    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.drawPoints(qp)
        qp.end()

    def getDataStart(self):
        self.receive = True
        self.adc_filename = f'{time.ctime()}'
        self.read_config = self.showCurrent()
        if bool(self.checkBox_4.checkState()):
            self.dataReceive.getcfgData(self.receive, f'{self.uart_list_comboBox.currentText()}', 'average')
        else:
            self.dataReceive.getcfgData(self.receive, f'{self.uart_list_comboBox.currentText()}', 'start')

    def clearOut(self):
        self.textEdit_2.clear()
        self.textEdit.clear()

    def showDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Путь до файла')[0]
        self.lineEdit_dac_file_path.setText(fname)

    def showDialog_2(self):
        fname = QFileDialog.getExistingDirectory(self, 'Директория для записи')
        self.lineEdit_adc_file_path.setText(fname)

    def drawPoints(self, qp):
    
        #for i in range(len(self.outimg)):
            #if (i + 1) < len(self.outimg):
                #qp.drawLine(self.outimg[i][0] + 143,self.outimg[i][1] + 75, self.outimg[i + 1][0] + 143, self.outimg[i + 1][1] + 75)
        
        qp.setPen(Qt.black)
        size = self.size()
        for i in range(631):
            for j in range(171):
                if j == 0 or j == 170 or j == 85 or i == 0 or i == 630:
                    #qp.setPen(Qt.black)
                    qp.drawPoint(i + 50, j + 31)
                    qp.drawPoint(i + 50, j + 260)
                #else:
                #    qp.setPen(Qt.white)
                #    qp.drawPoint(i + 50, j + 31)
                #    qp.drawPoint(i + 50, j + 260)
        if len(self.receive_arr) != 0:
            qp.setPen(QPen(Qt.blue, 2, Qt.SolidLine))
            size = self.size()
            
            scale_factor_1 = len(self.out_arr)//631
            scale_factor_2 = len(self.receive_arr)//631
            
            if (self.intype_comboBox.currentText() == 'text'):
                vertical_factor = 1*(2048/170)
                amplitude = 4095
                add = 200
            elif (self.intype_comboBox.currentText() == 'uint8_t'):
                vertical_factor = 1/(128/170)
                amplitude = 255
                add = 275
                
            for i in range(630):
                qp.drawLine(i+50, int(85*(self.receive_arr[int(i*scale_factor_2)])) + 345,  i+51, int(85*(self.receive_arr[int((i+1)*scale_factor_2)])) + 345)
                #qp.drawLine(i+50, int((self.receive_arr[i*scale_factor_2]/vertical_factor)) + add,  i+51, int((self.receive_arr[(i+1)*scale_factor_2]/vertical_factor)) + add)
                #if i == 10:
                    #print(self.receive_arr[i*scale_factor_2])

        if len(self.out_arr) != 0:
            #self.drawflag = False
            qp.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            size = self.size()
            scale_factor_1 = len(self.out_arr)/631
            
            for i in range(630):
                #if i % 2 == 0:
                qp.drawLine(i+50, int(-85*(self.out_arr[int(i*scale_factor_1)]/2)) + 116,  i+51, int(-85*(self.out_arr[int((i+1)*scale_factor_1)]/2)) + 116)
        self.update()

    def runGraph(self):
        try:
            scale = float(self.lineEdit_amplitude.text())
            buf_size = 256
            number_of_periods = int(self.lineEdit_number_of_periods.text())
            zeroed_periods = int(self.lineEdit_zeroed_periods.text())
            zero_after = int((buf_size/number_of_periods)*zeroed_periods)
            duty = float(self.lineEdit_duty.text())
            print(number_of_periods)
            print(zeroed_periods)
            print(zero_after)
            if self.waveform_comboBox.currentText() == 'sin':
                self.out_arr = []
                for i in range(buf_size):
                    if i < zero_after:
                        self.out_arr.append(scale*math.sin(((number_of_periods*2*math.pi)/buf_size)*i))
                    else:
                        self.out_arr.append(0)

            if self.waveform_comboBox.currentText() == 'triangle':
                self.out_arr = []
                for i in range(buf_size):
                    if i < zero_after:
                        self.out_arr.append(scale*(2/math.pi)*math.asin(math.sin(((number_of_periods*2*math.pi)/buf_size)*i)))
                    else:
                        self.out_arr.append(0)

            if self.waveform_comboBox.currentText() == 'square':
                
                self.out_arr = []
                for i in range(buf_size):
                    if i < zero_after:
                        if (math.atan(math.tan(((number_of_periods*math.pi)/buf_size)*i+ math.pi/2))) + math.pi/2 < duty*math.pi:
                            self.out_arr.append(scale*1)
                        else:
                            self.out_arr.append(0)
                    else:
                        self.out_arr.append(0)

            if self.waveform_comboBox.currentText() == 'sawtooth':
                self.out_arr = []
                for i in range(buf_size):
                    if i < zero_after:
                        self.out_arr.append(scale*(2/math.pi)*math.atan(math.tan(((number_of_periods*math.pi)/buf_size)*i)))
                    else:
                        self.out_arr.append(0)
                        
            if self.waveform_comboBox.currentText() == 'other':
                with open(f'{self.lineEdit_dac_file_path.text()}', "r") as file:
                    file_str = file.read()
                file_arr = []
                file_arr = file_str.split('\n')
                scale_factor = len(file_arr)/buf_size
                for i in range(buf_size):
                    self.out_arr.append(float(file_arr[int(i*scale_factor)]))
                print(len(self.out_arr))
                
                
            str = 'load'
            uart_arr = []
            uint8t_arr = []
            test_arr = []
            for i in range(len(self.out_arr)):
                uart_arr.append(int(((self.out_arr[i]+1)/2)*16383))
           
            for i in range(len(uart_arr)):
                try:
                    index = f'{i}'
                    if (len(index) < 3):
                        zero_str = '0'
                        zero_str *= 3 - len(index) 
                        index = zero_str + index
                    dac_val = f'{uart_arr[i]}'
                    if (len(dac_val) < 5):
                        zero_str = '0'
                        zero_str *= 5 - len(dac_val) 
                        dac_val = zero_str + dac_val
                    with serial.Serial(f'{self.uart_list_comboBox.currentText()}', 38400, timeout=3) as ser:
                        ser.write(f'load{index}_{dac_val}'.encode('utf-8'))
                        s = ser.read(10)
                        #print(s)
                        time.sleep(0.01)
                        self.textEdit.append(f'{s}'[2:-8])
                        self.textEdit_2.append(f'load{index}_{dac_val}')
                        
                except Exception as e:
                    print(e)
            
            self.drawflag = True
        except Exception as e:
            QMessageBox.critical(self, "Ошибка ", f"{e}", QMessageBox.Ok)
            
    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"piezo_oscilloscope", None))
        self.startButton.setText(QCoreApplication.translate("MainWindow", u"Начать", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Форма сигнала", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Периодов:", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"из", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Заполнение", None))
        self.pushButton_dacLoad.setText(QCoreApplication.translate("MainWindow", u"Загрузка", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Частота", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"дискретизации", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Прием:", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Передача:", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Принято:", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"Отправлено:", None))
        self.stopButton.setText(QCoreApplication.translate("MainWindow", u"Остановить", None))
        self.pushButton_dac_path_load.setText(QCoreApplication.translate("MainWindow", u"Файл", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Амплитуда", None))
        self.setButton.setText(QCoreApplication.translate("MainWindow", u"Настроить", None))
        self.curCfgButton.setText(QCoreApplication.translate("MainWindow", u"Текущ. конф.", None))
        self.checkBox.setText(QCoreApplication.translate("Dialog", u"Реле", None))
        self.checkBox_2.setText(QCoreApplication.translate("Dialog", u"Запись в файл", None))
        self.label_12.setText(QCoreApplication.translate("Dialog", u"Триггер:", None))
        self.pushButton_adc_path_load.setText(QCoreApplication.translate("MainWindow", u"Путь", None))
        self.pushButton_uart.setText(QCoreApplication.translate("MainWindow", u"Обн. UART", None))
        self.checkBox_3.setText(QCoreApplication.translate("Dialog", u"ЦАП Тр.", None))
        self.pushButton_clear.setText(QCoreApplication.translate("Dialog", u"Очистить", None))
        self.checkBox_4.setText(QCoreApplication.translate("Dialog", u"Усреднять", None))
        self.pushButton_plot.setText(QCoreApplication.translate("Dialog", u"pymatplot", None))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Ui_MainWindow()
    ex.setFixedSize(820, 676)
    ex.setupUi(ex)
    ex.show()
    sys.exit(app.exec_())