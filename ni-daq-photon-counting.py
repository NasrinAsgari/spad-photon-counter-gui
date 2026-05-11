import os
import sys
import datetime
import nidaqmx
import numpy as np
from nidaqmx.constants import Edge, AcquisitionType
from nidaqmx.stream_readers import CounterReader
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PhotonCounterGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Real-Time Photon Counter")
        self.setGeometry(100, 100, 1000, 400)

        # Configuration
        self.device = "Dev1"
        self.counter_channel = f"{self.device}/ctr0"
        self.clock_channel = f"{self.device}/Ctr1"
        self.sample_rate = 1000  # Set sample rate to 10 kHz for 100 µs bin time
        self.samples_per_read = 5000  # 5 seconds of data at 10 kHz
        self.data = np.zeros(self.samples_per_read, dtype=np.uint32)
        self.saving = False
        self.data_file = None
        self.meta_file = None
        self.trace_counter = 0

        # UI: Time trace plot
        self.canvas1 = FigureCanvas(Figure())
        self.ax1 = self.canvas1.figure.add_subplot(111)
        self.ax1.set_title("Photon Counts")
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Photon Counts")
        self.line1, = self.ax1.plot([], [], 'b-')

        # UI: Autocorrelation plot
        self.canvas2 = FigureCanvas(Figure(figsize=(3, 3)))
        self.ax2 = self.canvas2.figure.add_subplot(111)
        self.ax2.set_title("Autocorrelation")
        self.ax2.set_xlabel("Lag (s)")
        self.ax2.set_ylabel("ACF")
        self.line2, = self.ax2.plot([], [], 'r-')

        # UI: Buttons
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_acquisition)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setDisabled(True)
        self.stop_button.clicked.connect(self.stop_acquisition)

        self.save_button = QPushButton("Start Saving")
        self.save_button.clicked.connect(self.toggle_save)

        # Layouts
        plot_layout = QHBoxLayout()
        plot_layout.addWidget(self.canvas1)
        plot_layout.addWidget(self.canvas2)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.save_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(plot_layout)
        main_layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)

    def setup_tasks(self):
        self.clock_task = nidaqmx.Task()
        self.clock_task.co_channels.add_co_pulse_chan_freq(self.clock_channel, freq=self.sample_rate)
        self.clock_task.timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)

        self.count_task = nidaqmx.Task()
        self.count_task.ci_channels.add_ci_count_edges_chan(self.counter_channel, edge=Edge.RISING)
        self.count_task.timing.cfg_samp_clk_timing(rate=self.sample_rate,
                                                   source=f"/{self.device}/Ctr1InternalOutput",
                                                   sample_mode=AcquisitionType.CONTINUOUS)
        self.count_task.ci_channels.all.ci_count_edges_term = f"/{self.device}/PFI0"

        self.reader = CounterReader(self.count_task.in_stream)

    def start_acquisition(self):
        self.setup_tasks()
        self.clock_task.start()
        self.count_task.start()
        self.trace_counter = 0
        self.timer.start(10)  # ms
        self.start_button.setDisabled(True)
        self.stop_button.setEnabled(True)

    def update_plot(self):
        try:
            self.reader.read_many_sample_uint32(
                self.data,
                number_of_samples_per_channel=self.samples_per_read,
                timeout=5.0
            )

            diff = np.diff(self.data, prepend=self.data[0])
            time = np.arange(len(diff)) / self.sample_rate

            self.line1.set_data(time, diff)
            duration = self.samples_per_read / self.sample_rate
            self.ax1.set_xlim(0, duration)  # Adjust to 2 seconds for the duration
            self.ax1.set_ylim(0, max(10, np.max(diff) + 2))
            self.canvas1.draw()

            if self.saving and self.data_file:
                diff.tofile(self.data_file)

            self.trace_counter += 1
            if self.trace_counter % 2 == 0:  # autocorrelation every 2 traces
                self.update_autocorrelation(diff)

        except Exception as e:
            print("⚠️ Error during read:", e)
            self.stop_acquisition()

    def update_autocorrelation(self, signal):
        signal = signal - np.mean(signal)
        n = len(signal)
        acf = np.correlate(signal, signal, mode='full')[n - 1:]

        if acf[0] != 0:
            acf /= acf[0]
        else:
            return

        lag = np.arange(1, len(acf) + 1) / self.sample_rate
        lag_log = np.log10(lag + 1e-9)  # add offset to avoid log(0)

        self.line2.set_data(lag_log, acf)
        self.ax2.set_xlim(np.min(lag_log), np.max(lag_log))
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.canvas2.draw()

    def stop_acquisition(self):
        self.timer.stop()
        try:
            self.count_task.stop()
            self.clock_task.stop()
        except:
            pass
        try:
            self.count_task.close()
            self.clock_task.close()
        except:
            pass
        self.start_button.setEnabled(True)
        self.stop_button.setDisabled(True)

        if self.saving:
            if self.data_file:
                self.data_file.close()
            if self.meta_file:
                self.meta_file.close()

    def toggle_save(self):
        if self.saving:
            self.saving = False
            self.save_button.setText("Start Saving")
            if self.data_file:
                self.data_file.close()
            if self.meta_file:
                self.meta_file.close()
        else:
            self.saving = True
            self.save_button.setText("Stop Saving")

            today_str = datetime.date.today().strftime("%Y-%m-%d")
            base_dir = os.path.join("data", today_str)
            os.makedirs(base_dir, exist_ok=True)

            index = 0
            while True:
                bin_path = os.path.join(base_dir, f"measure_{index}.bin")
                txt_path = os.path.join(base_dir, f"info_measure_{index}.txt")
                if not os.path.exists(bin_path) and not os.path.exists(txt_path):
                    break
                index += 1

            self.data_file = open(bin_path, "wb")
            self.meta_file = open(txt_path, "w")

            self.meta_file.write(f"Start time: {datetime.datetime.now()}\n")
            self.meta_file.write(f"Sample rate: {self.sample_rate} Hz\n")
            #self.meta_file.write(f"Bin time: {1 / self.sample_rate * 1e6:.1f} µs\n")
            bin_time_sec = 1 / self.sample_rate
            trace_duration = self.samples_per_read * bin_time_sec
            self.meta_file.write(f"Bin time: {bin_time_sec * 1e6:.1f} µs\n")
            self.meta_file.write(f"Trace duration: {trace_duration:.3f} seconds\n")
            self.meta_file.write(f"Samples per read: {self.samples_per_read}\n")
            self.meta_file.write(f"Counter channel: {self.counter_channel}\n")
            self.meta_file.write(f"Clock channel: {self.clock_channel}\n")
            self.meta_file.flush()

            print(f"📁 Saving to folder: {base_dir}")
            print(f"🟢 Binary file: {bin_path}")
            print(f"📝 Info file:   {txt_path}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PhotonCounterGUI()
    window.show()
    sys.exit(app.exec_())
