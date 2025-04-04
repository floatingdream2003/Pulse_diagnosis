import numpy as np
import cv2
from scipy import signal
from sklearn.preprocessing import MinMaxScaler

# 全局变量
data_buffer = []  # 数据缓冲区
times = []        # 时间戳缓冲区
buffer_size = 100 # 缓冲区大小
mutation_threshold = 10  # 突变判断阈值
RPPG = []#这个是存储的总的RPPG信号
###############################################################
#TODO：输入图像和帧率fps
#TODO：加上上面的全局变量
#计算实时呼吸率的函数
def respiration_rate1(rois, fps):
    global data_buffer
    green = np.mean(np.mean(roi[:, :, 1]) for roi in rois)

#判断突变
    if len(RPPG) > 10:
        diff = abs(green - np.mean(RPPG[-min(10, len(RPPG)):]))
        if diff > mutation_threshold:
            print("RPPG 数据突变，使用历史值代替当前值...")
            green = RPPG[-1]

    RPPG.append(green)
    data_buffer.append(green)

    if len(data_buffer) < buffer_size:
        print("数据不足，等待缓冲区填满...")
        return None
    #TODO:这个None后续根据需求改为别的，指的是缓冲区没有满，未开始计算

    if len(data_buffer) > buffer_size:
        data_buffer = data_buffer[-buffer_size:]

    processed_data = signal.detrend(data_buffer)
    processed_data = normalized(processed_data, -0.1, 0.1)
    resp_data_filtered = butter_bandpass_filter(processed_data, 0.05, 0.1, fps, order=3)

    resp_fft = np.fft.fft(resp_data_filtered)
    resp_fft = np.abs(resp_fft)

    resp_peak, resp_n = find_fft_peak(resp_fft, fps)
    resp_freq = resp_peak / resp_n * fps

    resp_rate = resp_freq * 60
    print("呼吸率：", resp_rate)

    return resp_rate#实时计算的呼吸率

#计算历史总呼吸率的函数
def respiration_rate2(RPPG, fps):
    processed_data = normalized(RPPG, -0.1, 0.1)
    resp_data_filtered = butter_bandpass_filter(processed_data, 0.05, 0.1, fps, order=3)

    resp_fft = np.fft.fft(resp_data_filtered)
    resp_fft = np.abs(resp_fft)

    resp_peak, resp_n = find_fft_peak(resp_fft, fps)
    resp_freq = resp_peak / resp_n * fps

    resp_rate = resp_freq * 60
    print("呼吸率：", resp_rate)

    return resp_rate#总呼吸率（视频结束后，计算的一个总呼吸率）


def find_fft_peak(chdata_fft, fps):
    n = len(chdata_fft)

    lb = int(0.5 * n / fps)
    ub = int(4.0 * n / fps)

    fft_peak = np.argmax(chdata_fft[lb:ub]) + lb

    return fft_peak, n

def normalized(data, y_min, y_max):
    data = np.array(data).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(y_min, y_max))
    normalized_data = scaler.fit_transform(data)
    return normalized_data.flatten()


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

if __name__ == "__main__":
    a = cv2.imread('img.png')
    print(respiration_rate1(a, 30))