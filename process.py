import cv2
import numpy as np
import time
from scipy import signal
from signal_processing import Signal_processing
from scipy.signal import find_peaks

def wrist_detect(img):

    height, width, _ = img.shape
    start_x = width // 2 - 100
    start_y = height // 2 - 100
    end_x = width // 2 + 100
    end_y = height // 2 + 100

    # Recognize the rectangular area of the ROI
    # cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 255), 3)
    cv2.putText(img, 'PUT YOUR WRIST HERE', (start_x, start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    height = end_y - start_y
    part_height = height // 3

    roi_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
    square_size = (end_x - start_x) // 4

    # 平均分成三份
    roi_parts = []
    for i in range(1, 4):
       
        square_start_x = end_x - square_size
        square_start_y = start_y + part_height * (i-1)
        cv2.rectangle(img, (square_start_x, square_start_y), (square_start_x + square_size, square_start_y + square_size), roi_colors[i-1], 2)
        y = start_y + part_height * i
        cv2.line(img, (start_x, y), (end_x, y), (0, 255, 255), 3)
        roi = img[square_start_y:square_start_y + square_size, square_start_x:square_start_x + square_size]
        roi_parts.append(roi)

    cun_roi, guan_roi, chi_roi = roi_parts
    roi_names = ['cun', 'guan', 'chi']


    for i, img_roi in enumerate([cun_roi, guan_roi, chi_roi]):
        # Color space conversion
        img_HSV = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)
        HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255))
        HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        img_YCrCb = cv2.cvtColor(img_roi, cv2.COLOR_BGR2YCrCb)
        YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135))
        YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

        # Merge
        global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
        global_mask=cv2.medianBlur(global_mask,3)
        global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))

        # Find the largest contour
        contours, _ = cv2.findContours(global_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ROI = img_roi
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            contour_mask = np.zeros_like(global_mask)
            cv2.drawContours(contour_mask, [max_contour], -1, (255), thickness=cv2.FILLED)
            ROI = cv2.bitwise_and(img_roi, img_roi, mask=contour_mask)

        cv2.rectangle(img, (start_x, start_y + part_height * i), (end_x, start_y + part_height * (i + 1)), roi_colors[i], 3)
        
        # cv2.imshow(roi_names[i], ROI)

    return cun_roi, guan_roi, chi_roi

class Process(object):
    def __init__(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.buffer_size = 100
        self.times = [] 
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.bpms = []
        self.peaks = []
        self.sp = Signal_processing()
        self.signal = True
        self.T = 0

    def extractColor(self, frame):
        
        #r = np.mean(frame[:,:,0])
        g = np.mean(frame[:,:,1])
        #b = np.mean(frame[:,:,2])
        #return r, g, b
        return g


    def run(self,ROIs):
        frame = self.frame_in
        green_val = self.sp.extract_color(ROIs)
        self.frame_out = frame

        L = len(self.data_buffer)
        g = green_val # 从ROIS中提取绿色平均值（简写）

        if L == 0:
            self.t0 = time.time()

        if abs(g - np.mean(self.data_buffer)) > 10 and L > 99:
            g = self.data_buffer[-1]

        self.data_buffer.append(g)

        if L >= self.buffer_size:
            self.data_buffer = self.data_buffer[5:]

            if self.signal==True:
                t_total = time.time() - self.t0
                self.fps = self.buffer_size / t_total
                self.T = self.buffer_size / self.fps
                self.signal=False

            # 信号预处理
            processed = np.array(self.data_buffer)
            processed = signal.detrend(processed)  # 去除趋势
            norm = processed / np.max(np.abs(processed))  # 归一化到 [-1, 1]

            fft_processed = np.fft.rfft(norm)
            spectrum = np.abs(fft_processed)

            # 检测峰值（仅返回峰值数量）
            peaks, _ = find_peaks(
                spectrum,
                height=0.1 * np.max(spectrum),  # 高度阈值
                distance=5,  # 最小峰间距
            )
            numpeak = len(peaks)

            self.bpm = numpeak / self.T * 60
            self.bpms.append(self.bpm)
            print(f"bpm:{self.bpm}")
        return True
    
    def reset(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        #self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.times = [] 
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        #心率
        self.bpm = 0
        self.bpms = []
        
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y 
