import numpy as np
from PyQt5.QtGui import QFont, QImage, QPixmap,QPalette, QColor
from PyQt5.QtWidgets import QComboBox, QFileDialog, QStatusBar, QDesktopWidget, QMessageBox, QMainWindow, \
     QFrame, QApplication, QWidget, QPushButton, QLabel, QHBoxLayout, QVBoxLayout
import pyqtgraph as pg
import sys


from process import *
from webcam import Webcam
from video import Video
from interface import waitKey
from hr_analyze import total

# from HRV import heart_rate_heatplot, classify_pulse



class GUI(QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()

        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.screenheight = self.screenRect.height()
        self.screenwidth = self.screenRect.width()


        print("Screen height {}".format(self.screenheight))#1600
        print("Screen width {}".format(self.screenwidth))#2560


        self.initUI()
        self.webcam = Webcam()
        self.video = Video()
        self.input = self.webcam
        self.dirname = ""
        print("输入：实时拍摄")
        self.statusBar.showMessage("输入：实时拍摄", 5000)
        self.btnOpen.setEnabled(False)
        self.process = Process()
        self.status = False
        self.frame = np.zeros((10, 10, 3), np.uint8)
        # self.plot = np.zeros((10,10,3),np.uint8)
        self.bpm = 0
        self.terminate = False

        self.heart_rate = np.array([])
        self.nni_list = np.array([])  # 用于存储每次计算的 nni
        self.rpeaks_list = np.array([])  # 用于存储每次计算的 rpeaks

        # self.bpm_file_path = self.create_bpm_file_path()  # 创建心率数据文件的路径

    def initUI(self):

        # set font 字体
        font = QFont()
        font.setPointSize(16)

        # background——》》》灰色背景
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(230, 230, 230))  # 灰色的RGB值
        self.setPalette(palette)
#2560/1600
        # widgets 按钮
        self.btnStart = QPushButton("开始", self)
        self.btnStart.move(0, 0)
        self.btnStart.setFixedWidth(self.screenwidth*304/2560)
        self.btnStart.setFixedHeight(self.screenheight*50/1600)
        self.btnStart.setFont(font)
        self.btnStart.clicked.connect(self.run)

        self.btnOpen = QPushButton("打开文件", self)
        self.btnOpen.move(self.screenwidth*304/2560, 0)
        self.btnOpen.setFixedWidth(self.screenwidth*304/2560)
        self.btnOpen.setFixedHeight(self.screenheight*50/1600)
        self.btnOpen.setFont(font)
        self.btnOpen.clicked.connect(self.openFileDialog)#######################################################################111

        self.cbbInput = QComboBox(self)
        self.cbbInput.addItem("实时拍摄")
        self.cbbInput.addItem("本地视频")
        self.cbbInput.setCurrentIndex(0)
        self.cbbInput.setFixedWidth(self.screenwidth*304/2560)
        self.cbbInput.setFixedHeight(self.screenheight*50/1600)
        self.cbbInput.move(self.screenwidth*606/2560,self.screenheight*0/1600)
        self.cbbInput.setFont(font)
        self.cbbInput.activated.connect(self.selectInput)
        # -------------------

        # 创建字体对象
        font = QFont("Comic Sans MS", 15, QFont.Bold)

        # 设置标签
        # self.lblDisplay = QLabel(self)  # label to show frame from camera 用于显示摄像头帧的标签
        # self.lblDisplay.setGeometry(0, self.screenheight*50/1600, self.screenwidth*910/2560,self.screenheight*500/1600)
        # self.lblDisplay.setStyleSheet("background-color: #FFFFFF")
        # self.lblDisplay.setScaledContents(True)  # 让内容自动缩放以填充标签
        self.frame = QFrame(self)
        self.frame.setGeometry(0, self.screenheight*50/1600, self.screenwidth*910/2560,self.screenheight*500/1600)
        # self.frame.setStyleSheet("border: 2px solid black;")  # 添加黑色边框

        self.lblDisplay = QLabel(self.frame)
        self.lblDisplay.setGeometry(0, 0, self.screenwidth*910/2560,self.screenheight*500/1600)
        self.lblDisplay.setStyleSheet("background-color: #FFFFFF")
        self.lblDisplay.setScaledContents(True)

        self.lblHR_1 = QLabel(self)  # label to show HR change over time 用于显示随时间变化的心率的标签
        self.lblHR_1.setGeometry(self.screenwidth * 0 / 2560, self.screenheight * 600 / 1600, self.screenwidth * 300 / 2560,self.screenheight * 40 / 1600)
        self.lblHR_1.setFont(font)
        # self.lblHR_1.setText("实时心率（寸）: ")

        self.lblHR2_1 = QLabel(self)  # label to show stable HR 用于显示稳定心率的标签
        self.lblHR2_1.setGeometry(self.screenwidth * 0 / 2560, self.screenheight * 650 / 1600, self.screenwidth * 300 / 2560,self.screenheight * 40 / 1600)
        self.lblHR2_1.setFont(font)
        # self.lblHR2_1.setText("稳定心率（寸）: ")

        self.lblHR_2 = QLabel(self)  # label to show HR change over time 用于显示随时间变化的心率的标签
        self.lblHR_2.setGeometry(self.screenwidth * 0 / 2560, self.screenheight * 750 / 1600, self.screenwidth * 300 / 2560,self.screenheight * 40 / 1600)
        self.lblHR_2.setFont(font)
        # self.lblHR_2.setText("实时心率（关）: ")

        self.lblHR2_2 = QLabel(self)  # label to show stable HR 用于显示稳定心率的标签
        self.lblHR2_2.setGeometry(self.screenwidth * 0 / 2560, self.screenheight * 800 / 1600, self.screenwidth * 300 / 2560,self.screenheight * 40 / 1600)
        self.lblHR2_2.setFont(font)
        # self.lblHR2_2.setText("稳定心率（关）: ")

        self.lblHR_3 = QLabel(self)  # label to show HR change over time 用于显示随时间变化的心率的标签
        self.lblHR_3.setGeometry(self.screenwidth * 0 / 2560, self.screenheight * 900 / 1600, self.screenwidth * 300 / 2560,self.screenheight * 40 / 1600)
        self.lblHR_3.setFont(font)
        # self.lblHR_3.setText("实时心率（尺）: ")

        self.lblHR2_3 = QLabel(self)  # label to show stable HR 用于显示稳定心率的标签
        self.lblHR2_3.setGeometry(self.screenwidth*0/2560,self.screenheight*950/1600,self.screenwidth*300/2560,self.screenheight*40/1600)
        self.lblHR2_3.setFont(font)
        # self.lblHR2_3.setText("稳定心率（尺）: ")


        # dynamic plot 动态绘图
        self.signal_Plt1 = pg.PlotWidget(self)
        self.signal_Plt1.move(self.screenwidth*940/2560,self.screenheight*0/1600)
        self.signal_Plt1.resize(self.screenwidth*940/2560,self.screenheight*352/1600)
        self.signal_Plt1.setLabel('bottom', "心率图（寸）")
        self.signal_Plt1.setBackground((230, 230, 230))  # 设置背景为透明
        axis_x = self.signal_Plt1.getAxis('bottom')
        axis_x.setPen(pg.mkPen(color='black', width=2))
        axis_y = self.signal_Plt1.getAxis('left')
        axis_y.setPen(pg.mkPen(color='black', width=2))
        axis_x.setTextPen('black')
        axis_y.setTextPen('black')

        self.signal_Plt2 = pg.PlotWidget(self)
        self.signal_Plt2.move(self.screenwidth*940/2560,self.screenheight*352/1600)
        self.signal_Plt2.resize(self.screenwidth*940/2560,self.screenheight*352/1600)
        self.signal_Plt2.setLabel('bottom', "心率图（关）")
        self.signal_Plt2.setBackground((230, 230, 230))  # 设置背景为透明
        axis_x = self.signal_Plt2.getAxis('bottom')
        axis_x.setPen(pg.mkPen(color='black', width=2))
        axis_y = self.signal_Plt2.getAxis('left')
        axis_y.setPen(pg.mkPen(color='black', width=2))
        axis_x.setTextPen('black')
        axis_y.setTextPen('black')

        self.signal_Plt3 = pg.PlotWidget(self)
        self.signal_Plt3.move(self.screenwidth*940/2560,self.screenheight*704/1600)
        self.signal_Plt3.resize(self.screenwidth*940/2560,self.screenheight*352/1600)
        self.signal_Plt3.setLabel('bottom', "心率图（尺）")
        self.signal_Plt3.setBackground((230, 230, 230))  # 设置背景为透明
        axis_x = self.signal_Plt3.getAxis('bottom')
        axis_x.setPen(pg.mkPen(color='black', width=2))
        axis_y = self.signal_Plt3.getAxis('left')
        axis_y.setPen(pg.mkPen(color='black', width=2))
        axis_x.setTextPen('black')
        axis_y.setTextPen('black')

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(200)

        self.statusBar = QStatusBar()
        self.statusBar.setFont(font)
        self.setStatusBar(self.statusBar)

        # config main window 配置主窗口
        self.setGeometry(self.screenwidth*100/2560, self.screenheight*50/1600, self.screenwidth*1880/2560, self.screenheight*1056/1600)  # self.screenheight
        self.center()
        self.setWindowTitle("Heart rate monitor")


        ################Layout#################################
        btn_layout = QHBoxLayout()#横向布局
        btn_layout.addWidget(self.btnStart)
        btn_layout.addWidget(self.btnOpen)
        btn_layout.addWidget(self.cbbInput)

        camera_Layout = QVBoxLayout()#纵向布局
        camera_Layout.addLayout(btn_layout)
        camera_Layout.addWidget(self.lblDisplay)

        lbl_Layout = QVBoxLayout()
        lbl_Layout.addWidget(self.lblHR_1)
        lbl_Layout.addWidget(self.lblHR2_1)
        lbl_Layout.addWidget(self.lblHR_2)
        lbl_Layout.addWidget(self.lblHR2_2)
        lbl_Layout.addWidget(self.lblHR_3)
        lbl_Layout.addWidget(self.lblHR2_3)

        left_Layout = QVBoxLayout()
        left_Layout.addLayout(camera_Layout)
        left_Layout.addLayout(lbl_Layout)

        signal_Layout = QVBoxLayout()
        signal_Layout.addWidget(self.signal_Plt1)
        signal_Layout.addWidget(self.signal_Plt2)
        signal_Layout.addWidget(self.signal_Plt3)

        right_Layout = QVBoxLayout()
        right_Layout.addLayout(signal_Layout)

        main_Layout = QHBoxLayout()
        main_Layout.addLayout(left_Layout)
        main_Layout.addLayout(right_Layout)

        self.setGeometry = QWidget()
        self.setGeometry.setLayout(main_Layout)
        self.setCentralWidget(self.setGeometry)

        self.show()

    # 初始化用户界面

    def update(self):
        pen = pg.mkPen(color=(0, 0, 0), width=2)

        self.signal_Plt1.clear()
        self.signal_Plt1.plot(self.process.samples[20:], pen=pen)

        self.signal_Plt2.clear()
        self.signal_Plt2.plot(self.process.samples[20:], pen=pen)

        self.signal_Plt3.clear()
        self.signal_Plt3.plot(self.process.samples[20:], pen=pen)

    # 更新绘图数据，清除原有数据并绘制新数据。

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # 窗口居中显示

    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Message", "Are you sure want to quit",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            event.accept()
            self.input.stop()
            # cv2.destroyAllWindows()
            self.terminate = True
            sys.exit()

        else:
            event.ignore()

    # 处理关闭事件，弹出确认对话框并根据用户选择执行相应操作。

    def selectInput(self):
        self.reset()
        if self.cbbInput.currentIndex() == 0:
            self.input = self.webcam
            print("Input: webcam")
            self.btnOpen.setEnabled(False)
            # self.statusBar.showMessage("Input: webcam",5000)
        elif self.cbbInput.currentIndex() == 1:
            self.input = self.video
            print("Input: video")
            self.btnOpen.setEnabled(True)
            # self.statusBar.showMessage("Input: video",5000)

    # 根据下拉框选择的输入源重置界面并更新相应设置。

    def key_handler(self):
        self.pressed = waitKey(1) & 255  # wait for keypress for 10 ms
        if self.pressed == 27:  # exit program on 'esc'
            print("[INFO] Exiting")
            self.webcam.stop()
            sys.exit()

    # 摁esc退出

    def openFileDialog(self):
        self.dirname = QFileDialog.getOpenFileName(self, 'OpenFile')

    # 打开文件对话框

    def reset(self):
        self.process.reset()
        self.lblDisplay.clear()
        self.lblDisplay.setStyleSheet("background-color: #000000")

    # 重置界面
    def ROI(self):
        frame = self.input.get_frame()
        ROI1, ROI2, ROI3 = wrist_detect(frame)
        return ROI1, ROI2, ROI3

#将心率存储到数组中
    def hr_store(self,bpm,limit_high=90,limit_low=50):
        if bpm<=limit_high and bpm>=limit_low:
            self.heart_rate=np.append(self.heart_rate,bpm)

    # def nni123(self, bpm, sampling_rate=75, duration=10):
    #     if bpm <= 0:
    #         raise ValueError("Heart rate must be greater than 0.")
    #
    #     # 计算NN间期（毫秒）
    #     nni = 60000 / bpm  # 毫秒
    #
    #     # 计算R波峰时间
    #     num_beats = int(duration * 1000 / nni)  # 计算总的心跳数
    #     rpeaks = np.arange(0, num_beats * nni, nni)  # R波峰时间点
    #
    #     # 确保rpeaks不超过duration
    #     rpeaks = rpeaks[rpeaks < duration * 1000]
    #
    #     return nni, rpeaks


    def main_loop(self, ROI,lblHR,lblHR2):
        frame = self.input.get_frame()

        self.process.frame_in = frame
        if self.terminate == False:
            ret = self.process.run(ROI)  ####################

        # cv2.imshow("Processed", frame)
        if ret == True:
            self.frame = self.process.frame_out  # get the frame to show in GUI
            self.f_fr = ROI
            # print(self.f_fr.shape)
            self.bpm = self.process.bpm  # get the bpm change over the time
        else:
            self.frame = frame
            self.f_fr = np.zeros((10, 10, 3), np.uint8)
            self.bpm = 0


        # # 将心率转换为NNI和R波峰时间
        # if self.bpm > 0:
        #     nni, rpeaks = self.nni123(self.bpm)
        #     self.nni_list=np.append(self.nni_list,nni)  # 将新的 nni 添加到列表
        #     self.rpeaks_list=np.append(self.rpeaks_list,rpeaks)  # 将新的 rpeaks 添加到列表
        #


        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
        cv2.putText(self.frame, "FPS " + str(float("{:.2f}".format(self.process.fps))),
                    (20, 460), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
        img = QImage(self.frame, self.frame.shape[1], self.frame.shape[0],
                     self.frame.strides[0], QImage.Format_RGB888)
        self.lblDisplay.setPixmap(QPixmap.fromImage(img))

        self.f_fr = cv2.cvtColor(self.f_fr, cv2.COLOR_RGB2BGR)
        self.f_fr = np.transpose(self.f_fr, (0, 1, 2)).copy()
        f_img = QImage(self.f_fr, self.f_fr.shape[1], self.f_fr.shape[0],
                       self.f_fr.strides[0], QImage.Format_RGB888)

        lblHR.setText("Freq: " + str(float("{:.2f}".format(self.bpm))))
        # print("实时心率：", self.bpm)

        self.hr_store(self.bpm)


        #将心率信息存储到txt中########################################################################
        # if(self.bpm!=0):
        #     with open(self.bpm_file_path, "a") as file:
        #         file.write(f"{self.bpm} ")


        if self.process.bpms.__len__() > 50:
            if (max(self.process.bpms - np.mean(self.process.bpms)) < 5):  # show HR if it is stable -the change is not over 5 bpm- for 3s
                lblHR2.setText("Heart rate: " + str(float("{:.2f}".format(np.mean(self.process.bpms)))) + " bpm")
        self.key_handler()  # if not the GUI cant show anything

    def run(self, input):
        print("run")
        self.reset()
        input = self.input
        self.input.dirname = self.dirname
        if self.input.dirname == "" and self.input == self.video:
            print("请先选择一个视频")
            # self.statusBar.showMessage("choose a video first",5000)
            return
        if self.status == False:
            self.status = True
            input.start()
            self.btnStart.setText("停止")
            self.cbbInput.setEnabled(False)
            self.btnOpen.setEnabled(False)
            self.lblHR2_1.clear()
            self.lblHR2_2.clear()
            self.lblHR2_3.clear()
            while self.status == True:
                ROI1, ROI2, ROI3 = self.ROI()
                # print(ROI1)
                self.main_loop(ROI1,self.lblHR_1,self.lblHR2_1)
                self.main_loop(ROI2,self.lblHR_2,self.lblHR2_2)
                self.main_loop(ROI3,self.lblHR_3,self.lblHR2_3)

        elif self.status == True:
            self.status = False
            input.stop()
            self.btnStart.setText("开始")
            self.cbbInput.setEnabled(True)

            # print(self.heart_rate)
            total(self.heart_rate)


            # heart_rate_heatplot(self.nni_list, self.rpeaks_list)
            # persentage=classify_pulse(self.nni_list)
            # print(persentage)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GUI()
    sys.exit(app.exec_())
