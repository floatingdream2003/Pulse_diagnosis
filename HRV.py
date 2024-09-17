import os
import sys
import warnings
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime as dt
from matplotlib.projections import register_projection

# BioSPPy imports
import biosppy

# Local imports
import pyhrv
import pyhrv.time_domain
import pyhrv.frequency_domain
import pyhrv.nonlinear

def heart_rate_heatplot(nni=None,
						rpeaks=None,
						signal=None,
						sampling_rate=1000.,
						age=18,
						gender='male',
						interval=None,
						figsize=None,
						show=True):

	# Helper function
	def _get_classification(val, data):
		for key in data.keys():
			if data[key][0] <= int(val) <= data[key][1]:
				return key

	# Check input
	if signal is not None:
		rpeaks = biosppy.signals.ecg.ecg(signal=signal, sampling_rate=sampling_rate, show=False)[2]
	elif nni is None and rpeaks is None:
		raise TypeError('No input data provided. Please specify input data.')

	# Get NNI series
	nn = pyhrv.utils.check_input(nni, rpeaks)

	# Compute HR data and
	hr_data = 60000/nni
	t = np.cumsum(nn) / 1000
	interval = pyhrv.utils.check_interval(interval, limits=[0, t[-1]], default=[0, t[-1]])

	# Prepare figure
	if figsize is None:
		figsize = (12, 5)
	fig, (ax, ax1, ax2) = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [12, 1, 1]})
	ax1.axis("off")
	fig.suptitle("Heart Rate Heat Plot (%s, %s)" % (gender, age))

	# X-Axis configuration
	# Set x-axis format to seconds if the duration of the signal <= 60s
	if interval[1] <= 60:
		ax.set_xlabel('Time [s]')
	# Set x-axis format to MM:SS if the duration of the signal > 60s and <= 1h
	elif 60 < interval[1] <= 3600:
		ax.set_xlabel('Time [MM:SS]')
		formatter = mpl.ticker.FuncFormatter(lambda ms, x: str(dt.timedelta(seconds=ms))[2:])
		ax.xaxis.set_major_formatter(formatter)
	# Set x-axis format to HH:MM:SS if the duration of the signal > 1h
	else:
		ax.set_xlabel('Time [HH:MM:SS]')
		formatter = mpl.ticker.FuncFormatter(lambda ms, x: str(dt.timedelta(seconds=ms)))
		ax.xaxis.set_major_formatter(formatter)

	# Set gender
	if gender not in ["male", "m", "female", "f"]:
		raise ValueError("Unknown gender '%s' for this database." % gender)
	else:
		if gender == 'm':
			gender = 'male'
		elif gender == 'f':
			gender = 'female'

	# Load comparison data from database
	database = json.load(open(os.path.join(os.path.split(__file__)[0], './hr_heatplot.json')))

	# Get database values
	if age > 17:
		for key in database["ages"].keys():
			if database["ages"][key][0] - 1 < age < database["ages"][key][1] + 1:
				_age = database["ages"][key][0]

		color_map = database["colors"]
		data = database[gender][str(_age)]
		order = database["order"]

		# Plot with information based on reference database:
		# Create classifier counter (preparation for steps after the plot)
		classifier_counter = {}
		for key in data.keys():
			classifier_counter[key] = 0

		# Add threshold lines based on the comparison data
		for threshold in data.keys():
			ax.hlines(data[threshold][0], 0, t[-1], linewidth=0.4, alpha=1, color=color_map[threshold])
		ax.plot(t, hr_data, 'k--', linewidth=0.5)

		# Add colorized HR markers
		old_classifier = _get_classification(hr_data[0], data)
		start_index = 0
		end_index = 0
		for hr_val in hr_data:
			classifier_counter[old_classifier] += 1
			current_classifier = _get_classification(hr_val, data)
			if current_classifier != old_classifier:
				ax.plot(t[start_index:end_index], hr_data[start_index:end_index], 'o',
						markerfacecolor=color_map[old_classifier], markeredgecolor=color_map[old_classifier])
				start_index = end_index
				old_classifier = current_classifier
			end_index += 1

		# Compute distribution of HR values in %
		percentages = {}
		_left = 0
		legend = []
		for i in range(7):
			classifier = str(order[str(i)][0])
			percentages[classifier] = float(classifier_counter[classifier]) / hr_data.size * 100
			ax2.barh(0, percentages[classifier], left=_left, color=color_map[classifier])
			_left += percentages[classifier]
			print(percentages[classifier],",")
			legend.append(mpl.patches.Patch(label="%s\n(%.2f%s)" % (order[str(i)][1], percentages[classifier], "$\%$"),
											fc=color_map[classifier]))
			ax2.get_yaxis().set_visible(False)
		ax.legend(handles=legend, loc=8, ncol=7)
	elif age <= 0:
		raise ValueError("Age cannot be <= 0.")
	else:
		warnings.warn("No reference data for age %i available." % age)
		ax.plot(t, hr_data, 'k--', linewidth=0.5)
		ax2.plot("", 0)

	# Set axis limits
	ax.axis([interval[0], interval[1], hr_data.min() * 0.7, hr_data.max() * 1.1])
	ax.set_ylabel('Heart Rate [$1/min$]')
	ax2.set_xlim([0, 100])
	ax2.set_xlabel("Distribution of HR over the HR classifiers [$\%$]")

	# Show plot
	if show:
		plt.show()

	# Output
	return biosppy.utils.ReturnTuple((fig, ), ('hr_heatplot', ))


def classify_pulse(nni_array):
    # 计算心率数据的标准差
    std_dev = np.std(nni_array)

    # 计算心率
    heart_rate = 60000 / np.mean(nni_array)  # 每分钟的跳动次数

    # 计算心率的波动
    fluctuation = np.std(nni_array) / np.mean(nni_array)

    # 计算心率的变异系数
    cv = std_dev / np.mean(nni_array)

    # 判断脉象
    pulse_types = {
        "浮脉": "浮脉" if cv > 0.1 else "沉脉",
        "沉脉": "沉脉" if cv <= 0.1 else "浮脉",
        "迟脉": "迟脉" if heart_rate < 15 else "数脉",
        "数脉": "数脉" if heart_rate >= 12 else "迟脉",
        "虚脉": "虚脉" if fluctuation > 0.1 else "实脉",
        "实脉": "实脉" if fluctuation <= 0.1 else "虚脉"
    }

    # 计算脉象的百分比
    pulse_counts = {k: 0 for k in pulse_types.keys()}
    for nni in nni_array:
        heart_rate = 60000 / nni
        fluctuation = np.std(nni_array) / np.mean(nni_array)
        cv = std_dev / np.mean(nni_array)

        if cv > 0.1:
            pulse_counts["浮脉"] += 1
        else:
            pulse_counts["沉脉"] += 1

        if heart_rate < 15:
            pulse_counts["迟脉"] += 1
        else:
            pulse_counts["数脉"] += 1

        if fluctuation > 0.1:
            pulse_counts["虚脉"] += 1
        else:
            pulse_counts["实脉"] += 1

    total_pulses = len(nni_array)
    pulse_percentages = {k: (v / total_pulses) * 100 for k, v in pulse_counts.items()}

    return pulse_percentages
