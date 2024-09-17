import numpy as np

heart=np.array([65, 68, 72, 74, 69, 66, 71, 75, 70, 74,
67, 73, 76, 68, 72, 69, 74, 77, 65, 71,
78, 70, 72, 67, 73, 76, 68, 69, 74, 75,
72, 67, 78, 70, 74, 69, 73, 68, 75, 76,
71, 72, 67, 78, 70, 73, 66, 74, 69, 75,
72, 77, 68, 74, 71, 70, 73, 78, 67, 75,
72, 68, 76, 69, 71, 74, 78, 70, 65, 73,
77, 72, 74, 68, 75, 69, 71, 76, 73, 70,
77, 65, 74, 72, 68, 70, 73, 76, 71, 78,
67, 74, 69, 75, 72, 68, 77, 71, 74, 70,
73, 67, 76, 75, 72, 68, 74, 71, 78, 69,
73, 70, 67, 74, 75, 72, 78, 71, 66, 74,
69, 73, 76, 70, 72, 67, 75, 74, 68, 71,
78, 70, 73, 66, 74, 77, 72, 69, 75, 68,
])#测试数据

def chi_shu(heart_rate):
    counter_shu=0
    counter_chi=0
    normal_hr= 68
    for hr in heart_rate:
        if hr<=normal_hr:
            counter_chi+=1
        else:
            counter_shu+=1
    return counter_shu,counter_chi

def chen_fu(heart_rate):
    mean_hr=np.mean(heart_rate)
    counter_fu=0
    counter_chen=0
    for hr in heart_rate:
        if abs(hr - mean_hr)>3:
            counter_fu+=1
        else:
            counter_chen+=1
    return counter_fu,counter_chen


def analyze_segment(segment):
    threshold_std_dev = 1.5  # 示例阈值
    # 计算标准差
    std_dev = np.std(segment)
    # 判断是否为虚脉或实脉
    if std_dev > threshold_std_dev:
        return 1
    else:
        return 0
    #1虚脉
    #0实脉


def xu_shi(heart_rate,segment_size=5):
    counter_xu=0
    counter_shi=0
    # 计算可以处理的完整数据段的数量
    num_segments = len(heart_rate) // segment_size
    # 截断数据，只保留可以被 segment_size 整除的部分
    truncated_data = heart_rate[:num_segments * segment_size]

    for i in range(num_segments):
        segment = truncated_data[i * segment_size:(i + 1) * segment_size]
        result=analyze_segment(segment)
        if result == 1:
            counter_xu+=segment_size
        else:
            counter_shi+=segment_size
    return counter_xu,counter_shi

def percentage(chi,shu,chen,fu,xu,shi):
    total=chi+shu+chen+fu+xu+shi

    chi_percent = (chi / total) * 100
    shu_percent = (shu / total) * 100
    chen_percent = (chen / total) * 100
    fu_percent = (fu / total) * 100
    xu_percent = (xu / total) * 100
    shi_percent = (shi / total) * 100

    return chi_percent, shu_percent, chen_percent, fu_percent, xu_percent, shi_percent


def total(heart):
    chi, shu = chi_shu(heart)
    chen, fu = chen_fu(heart)
    xu, shi = xu_shi(heart)
    chi_percent, shu_percent, chen_percent, fu_percent, xu_percent, shi_percent = percentage(chi, shu, chen, fu, xu,
                                                                                             shi)

    print(f"Chi: {chi_percent:.2f}%")
    print(f"Shu: {shu_percent:.2f}%")
    print(f"Chen: {chen_percent:.2f}%")
    print(f"Fu: {fu_percent:.2f}%")
    print(f"Xu: {xu_percent:.2f}%")
    print(f"Shi: {shi_percent:.2f}%")


chi,shu=chi_shu(heart)
chen,fu=chen_fu(heart)
xu,shi=xu_shi(heart)
chi_percent, shu_percent, chen_percent, fu_percent, xu_percent, shi_percent=percentage(chi, shu, chen, fu, xu, shi)

print(f"Chi: {chi_percent:.2f}%")
print(f"Shu: {shu_percent:.2f}%")
print(f"Chen: {chen_percent:.2f}%")
print(f"Fu: {fu_percent:.2f}%")
print(f"Xu: {xu_percent:.2f}%")
print(f"Shi: {shi_percent:.2f}%")