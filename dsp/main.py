import struct
import pyaudio
import sys
import time
import wave

import numpy as np
from scipy.fftpack import fft
import scipy.signal

import pyqtgraph as pg

import config

import dsp
from dsp.qt import *


class Microphone:
    def __init__(self, rate=44100, channels=1, frames_per_buffer=1024, format=pyaudio.paInt16):
        self.rate = rate
        self.channels = channels
        self.frames_per_buffer = frames_per_buffer
        self.format = format
        self.sample_size = pyaudio.get_sample_size(format)
        self.framesize = self.sample_size * self.channels
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=format,
                                  channels=channels,
                                  rate=rate,
                                  input=True,
                                  # input_device_index=self.find_input_device(),
                                  frames_per_buffer=frames_per_buffer)

    def read(self, frames=None):
        if frames is None:
            frames = self.frames_per_buffer
        return self.stream.read(frames)

    def find_input_device(self):
        device_index = None
        for i in range(self.p.get_device_count()):
            devinfo = self.p.get_device_info_by_index(i)
            if devinfo["name"].lower() in ["mic", "input"]:
                device_index = i
        return device_index

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


class Recorder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.data = []

    def add_frames(self, data):
        self.data.extend(data)


class Waveform(pg.PlotItem):
    def __init__(self, settings, framesize):
        self.settings = settings
        self.framesize = framesize

        xlabels = [(0, '0'), (2048, '2048'), (4096, '4096')]
        xaxis = pg.AxisItem(orientation='bottom')
        xaxis.setTicks([xlabels])

        ylabels = [(0, '0'), (127, '128'), (255, '255')]
        yaxis = pg.AxisItem(orientation='left')
        yaxis.setTicks([ylabels])

        super().__init__(title='Waveform', axisItems=dict(bottom=xaxis, left=yaxis))

        self.traces = self.plot(pen='c', width=3)

        self.settings.interval_changed.connect(self.setup)
        self.setup()

    def setup(self):
        self.x = np.arange(0, self.framesize * self.settings.chunksize, 2)
        self.setYRange(0, 255, padding=0)
        self.setXRange(0, 2 * self.settings.chunksize, padding=0.005)

    def update(self, data):
        data = struct.unpack(str(self.framesize * self.settings.chunksize) + 'B', data)
        data = np.array(data, dtype='b')[::2] + 128
        self.traces.setData(self.x, data)


class Waveform2(pg.PlotItem):
    def __init__(self, settings):
        super().__init__(title='Waveform2')
        self.plotitem = self.plot()
        self.setYRange(-5000, 5000)
        self.setXRange(0, 1000)

    def update(self, data):
        data = np.frombuffer(data, np.int16)
        self.plotitem.setData(x=np.arange(len(data)), y=data)


class FFT(pg.PlotItem):
    def __init__(self, settings):
        super().__init__(title='FFT')
        self.plotitem = self.plot()
        self.setYRange(-100, 100)
        self.setXRange(0, 5000)

    def update(self, data):
        data = np.frombuffer(data, np.int16)
        dfft = 10 * np.log10(abs(np.fft.rfft(data)))
        self.plotitem.setData(x=np.arange(len(dfft))*10, y=dfft)


class Spectrogram(pg.PlotItem):
    def __init__(self, settings, framesize):
        self.settings = settings
        self.framesize = framesize

        super().__init__(title='Spectrum')

        self.settings.interval_changed.connect(self.setup)
        self.traces = self.plot(pen='m', width=3)
        self.setup()

    def setup(self):
        self.x = np.linspace(0, self.settings.rate / 2, self.settings.chunksize // 2)
        self.setLogMode(x=False, y=True)
        self.setYRange(-4, 0, padding=0)
        #self.setXRange(np.log10(20), np.log10(self.rate / 2), padding=0.005)

    def update(self, data):
        data = struct.unpack(str(self.framesize * self.settings.chunksize) + 'B', data)
        data = np.array(data, dtype='b')[::2] + 128
        data = fft(np.array(data, dtype='int8') - 128)
        data = np.abs(data[0:int(self.settings.chunksize / 2)]
                         ) * 2 / (128 * self.settings.chunksize)
        self.traces.setData(self.x, data)


class Spectrogram2(pg.PlotItem):
    def __init__(self, settings):
        super().__init__(title='Spectrum2')
        self.settings = settings
        self.settings.interval_changed.connect(self.setup)

        self.img = pg.ImageItem()
        self.addItem(self.img)

        self.setup()

    def setup(self):
        self.img_array = np.zeros((1000, self.settings.chunksize//2+1))

        # bipolar colormap
        pos = np.array([0., 1., 0.5, 0.25, 0.75])
        color = np.array([[0,255,255,255], [255,255,0,255], [0,0,0,255], (0, 0, 255, 255), (255, 0, 0, 255)], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        lut = cmap.getLookupTable(0.0, 1.0, 256)

        # set colormap
        self.img.setLookupTable(lut)
        self.img.setLevels([-50,40])

        # setup the correct scaling for y-axis
        freq = np.arange((self.settings.chunksize/2)+1)/(float(self.settings.chunksize)/self.settings.rate)
        yscale = 1.0/(self.img_array.shape[1]/freq[-1])
        self.img.scale((1./self.settings.rate)*self.settings.chunksize, yscale)

        self.setLabel('left', 'Frequency', units='Hz')

        # prepare window for later use
        self.win = np.hanning(self.settings.chunksize)

    def update(self, data):
        data = np.frombuffer(data, 'int16')

        # normalized, windowed frequencies in data chunk
        spec = np.fft.rfft(data*self.win) / self.settings.chunksize
        # get magnitude 
        psd = abs(spec)
        # convert to dB scale
        psd = 20 * np.log10(psd)

        # roll down one and replace leading edge with new data
        self.img_array = np.roll(self.img_array, -1, 0)
        self.img_array[-1:] = psd

        self.img.setImage(self.img_array, autoLevels=False)


class Spectrogram3(pg.PlotItem):
    def __init__(self, settings, channel=None):
        self.settings = settings
        self.channel = channel

        super().__init__(title='Spectrum3')
        self.setMouseEnabled(y=False)
        self.setYRange(0, 1000)
        self.setXRange(-self.settings.rate/2, self.settings.rate/2, padding=0)

    def update(self, data):
        shorts = struct.unpack("%dh" % (len(data) / 2), data)

        if self.settings.channels == 1:
            data = np.array(shorts)
        else:
            l = shorts[::2]
            r = shorts[1::2]
            data = np.array(l) if self.channel == 'left' else np.array(r)

        T = 1.0 / self.settings.rate
        N = data.shape[0]
        Pxx = (1./N)*np.fft.fft(data)
        f = np.fft.fftfreq(N, T)
        Pxx = np.fft.fftshift(Pxx)
        f = np.fft.fftshift(f)

        self.plot(x=f.tolist(), y=np.absolute(Pxx).tolist(), clear=True)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return scipy.signal.butter(order, [low, high], analog=False, btype='band', output='sos')


class ButterworthBandpass(pg.GraphicsLayout):
    def __init__(self, settings):
        self.settings = settings
        super().__init__()

        p1 = self.addPlot(row=0, col=0, title='Filtered Signal')
        p1.setLabel('bottom', 'time', units='s')
        p1.setYRange(-1000, 1000, padding=0)
        self.plotfiltered = p1.plot()

        p2 = self.addPlot(row=0, col=1, title='Frequency Response')
        p2.setLabel('bottom', 'Normalized frequency (1.0 = Nyquist)')
        p2.setLabel('left', 'Gain', units='dB')
        self.plotfreqz = p2.plot()


    def update(self, data):
        T = self.settings.interval
        lowcut = self.settings.lowcut
        highcut = self.settings.highcut
        x = np.linspace(0, T, self.settings.chunksize, endpoint=False)
        data = np.frombuffer(data, dtype=np.int16)
        sos = butter_bandpass(lowcut, highcut, self.settings.rate, order=6)
        y = scipy.signal.sosfilt(sos, data)
        self.plotfiltered.setData(x, y)

        w, h = scipy.signal.sosfreqz(sos, worN=2000)
        self.plotfreqz.setData(x=w/np.pi, y=20*np.log10(np.maximum(np.abs(h), 1e-5)))


class GraphicsWidget(pg.GraphicsLayoutWidget):
    def __init__(self, settings, mic):
        super().__init__()

        self.settings = settings
        self.settings.interval_changed.connect(self.update_interval)

        self.mic = mic
        self.traces = dict()

        self.plots = [
            Waveform(self.settings, self.mic.framesize),
            Waveform2(self.settings),
            FFT(self.settings),
            Spectrogram(self.settings, self.mic.framesize),
            Spectrogram2(self.settings),
            Spectrogram3(self.settings),
            ButterworthBandpass(self.settings)
        ]

        for i, plot in enumerate(self.plots):
            self.addItem(plot, row=i, col=1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.update_interval(self.settings.interval)
        self.timer.start()

    def update_interval(self, value):
        self.timer.setInterval(int(1000 * value))  # QTimer expects ms

    def stop(self):
        self.timer.stop()

    def update(self):
        data = self.mic.read(self.settings.chunksize)
        for plot in self.plots:
            plot.update(data)


class Sidebar(QWidget):
    interval_changed = pyqtSignal(float)

    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('Interval (s):'))
        self.spin_interval = pg.SpinBox(compactHeight=False, bounds=(0.01, 10.0))
        self.spin_interval.setValue(self.settings.interval)
        self.spin_interval.valueChanged.connect(self.interval_changed)
        hbox.addWidget(self.spin_interval)
        vbox.addLayout(hbox)
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('lowcut:'))
        self.spin_lowcut = pg.SpinBox(compactHeight=False, bounds=(0, 10000), step=10)
        self.spin_lowcut.setValue(self.settings.lowcut)
        self.spin_lowcut.valueChanged.connect(self.lowcut_changed)
        hbox.addWidget(self.spin_lowcut)
        hbox.addWidget(QLabel('highcut:'))
        self.spin_highcut = pg.SpinBox(compactHeight=False, bounds=(0, 10000), step=10)
        self.spin_highcut.setValue(self.settings.highcut)
        self.spin_highcut.valueChanged.connect(self.highcut_changed)
        hbox.addWidget(self.spin_highcut)
        vbox.addLayout(hbox)
        vbox.addStretch()
        self.setLayout(vbox)

    def interval_changed(self, value):
        self.settings.interval = value

    def lowcut_changed(self, value):
        self.settings.lowcut = value

    def highcut_changed(self, value):
        self.settings.highcut = value


class Settings(QObject):
    interval_changed = pyqtSignal(float)
    cutoff_changed = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.rate = 44100
        self.channels = 1
        self.interval = float(QSettings().value('interval') or 0.05)
        self.lowcut = float(QSettings().value('lowcut') or 500)
        self.highcut = float(QSettings().value('highcut') or 1250)

    @property
    def lowcut(self):
        return self._lowcut

    @lowcut.setter
    def lowcut(self, value):
        self._lowcut = value
        QSettings().setValue('lowcut', value)
        self.cutoff_changed.emit()

    @property
    def highcut(self):
        return self._highcut

    @highcut.setter
    def highcut(self, value):
        self._highcut = value
        QSettings().setValue('highcut', value)
        self.cutoff_changed.emit()

    @property
    def interval(self):
        return self._interval

    @interval.setter
    def interval(self, value):
        self._interval = value
        QSettings().setValue('interval', value)
        self.chunksize = int(self.rate * value)
        self.interval_changed.emit(value)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.app_name)
        self.settings = Settings()
        self.mic = Microphone(rate=self.settings.rate, channels=self.settings.channels)
        self.setup_ui()

    def setup_ui(self):
        splitter = QSplitter()
        self.graphics_widget = GraphicsWidget(self.settings, self.mic)
        splitter.addWidget(self.graphics_widget)
        self.sidebar = Sidebar(self.settings)
        splitter.addWidget(self.sidebar)
        self.setCentralWidget(splitter)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        self.menu = self.menuBar()
        m = self.menu.addMenu('&File')

        a = QAction("&Quit", self)
        a.setShortcut("Ctrl+Q")
        a.setStatusTip('Leave The App')
        a.setIcon(QIcon('pics/quit.png'))
        a.triggered.connect(self.close)
        m.addAction(a)

        about_action = QAction('&About ' + config.app_name, self)
        about_action.setStatusTip('About ' + config.app_name)
        about_action.triggered.connect(self.about)
        about_action.setShortcut('F1')

        help_menu = self.menu.addMenu('&Help')
        help_menu.addAction(about_action)

        live_action = QAction(QIcon('pics/live.png'), 'live', self)
        live_action.triggered.connect(self.live_triggered)

        record_action = QAction(QIcon('pics/record.png'), 'record', self)
        record_action.triggered.connect(self.record_triggered)

        save_action = QAction(QIcon('pics/save.png'), 'save', self)
        save_action.triggered.connect(self.save_triggered)

        open_action = QAction(QIcon('pics/open.png'), 'open', self)
        open_action.triggered.connect(self.open_triggered)

        stop_action = QAction(QIcon('pics/stop.png'), 'stop', self)
        stop_action.triggered.connect(self.stop_triggered)

        self.toolbar = self.addToolBar('Control')
        self.toolbar.setMovable(QSettings().value('control-toolbar-movable', type=bool) or False)
        self.toolbar.addAction(live_action)
        self.toolbar.addAction(record_action)
        self.toolbar.addAction(stop_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(open_action)
        self.toolbar.addAction(save_action)

    def createPopupMenu(self):
        menu = super().createPopupMenu()
        menu.addSeparator()
        menu.addAction('Toggle fixed', self.toggle_toolbar_fixed)
        return menu

    def toggle_toolbar_fixed(self):
        v = not self.toolbar.isMovable()
        QSettings().setValue('control-toolbar-movable', v)
        self.toolbar.setMovable(v)

    def live_triggered(self):
        pass

    def record_triggered(self):
        pass

    def stop_triggered(self):
        pass

    def save_triggered(self):
        pass

    def open_triggered(self):
        pass

    def closeEvent(self, event):
        self.graphics_widget.stop()
        self.mic.close()
        super().closeEvent(event)

    def about(self):
        date = '2020'
        QMessageBox.about(self, 'About ' + config.app_name,
            """
            <b>%s</b>
            <p>Python Digital Signal Processing</p>
            <p><table border="0" width="150">
            <tr>
            <td>Version:</td>
            <td>%s</td>
            </tr>
            <tr>
            <td>Date:</td>
            <td>%s</td>
            </tr>
            </table></p>
            """ % (config.app_name, dsp.__version__, date))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName(config.app_name)
    app.setOrganizationName(config.app_organization)
    app.setOrganizationDomain(config.app_url)

    pg.setConfigOptions(antialias=True)

    w = MainWindow()
    w.setWindowIcon(QIcon('pics/logo.png'))
    w.show()

    app.exec_()
    app.deleteLater()
    sys.exit()
