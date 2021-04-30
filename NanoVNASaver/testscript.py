from .Hardware.Hardware import Interface, get_VNA
from .Hardware.VNA import VNA
from .Calibration import Calibration
from NanoVNASaver.RFTools import Datapoint
from .Touchstone import Touchstone

from time import sleep
from typing import List, Tuple, NamedTuple

import numpy as np
import matplotlib.pyplot as plt


import logging

logger = logging.getLogger("testscript")

def truncate(values: List[List[Tuple]], count: int) -> List[List[Tuple]]:
    """truncate drops extrema from data list if averaging is active"""
    keep = len(values) - count
    logger.debug("Truncating from %d values to %d", len(values), keep)
    if count < 1 or keep < 1:
        logger.info("Not doing illegal truncate")
        return values
    truncated = []
    for valueset in np.swapaxes(values, 0, 1).tolist():
        avg = complex(*np.average(valueset, 0))
        truncated.append(
            sorted(valueset,
                   key=lambda v, a=avg:
                   abs(a - complex(*v)))[:keep])
    return np.swapaxes(truncated, 0, 1).tolist()

class MyTest():
    def __init__(self):
        self.interface = Interface("serial", "None")
        self.interface.setPort('/dev/ttyACM0')
        self.vna = VNA(self.interface)
        self.calibration = Calibration()

    def connect_device(self):
        if not self.interface:
            return
        with self.interface.lock:
            #self.interface = self.serialPortInput.currentData()
            logger.info("Connection %s", self.interface)
            try:
                self.interface.open()
                self.interface.timeout = 0.05
            except (IOError, AttributeError) as exc:
                logger.error("Tried to open %s and failed: %s",
                             self.interface, exc)
                return
            if not self.interface.isOpen():
                logger.error("Unable to open port %s", self.interface)
                return
        sleep(0.1)
        try:
            self.vna = get_VNA(self.interface)
        except IOError as exc:
            logger.error("Unable to connect to VNA: %s", exc)

        #self.vna.validateInput = self.settings.value("SerialInputValidation", True, bool)

        # connected
        logger.info("connected to VNA")

        frequencies = self.vna.readFrequencies()
        if not frequencies:
            logger.warning("No frequencies read")
            return
        logger.info("Read starting frequency %s and end frequency %s",
                    frequencies[0], frequencies[-1])

        #logger.debug("Starting initial sweep")
        #self.sweep_start()

    def disconnect_device(self):
        with self.interface.lock:
            logger.info("Closing connection to %s", self.interface)
            self.interface.close()

    def readAveragedSegment(self, start, stop, averages=1, truncates=0):
        values11 = []
        values21 = []
        freq = []
        logger.info("Reading from %d to %d. Averaging %d values",
                    start, stop, averages)
        for i in range(averages):
            logger.debug("Reading average no %d / %d", i+1, averages)
            freq, tmp11, tmp21 = self.readSegment(start, stop)
            values11.append(tmp11)
            values21.append(tmp21)

        if not values11:
            raise IOError("Invalid data during swwep")

        if truncates > 0 and averages > 1:
            logger.debug("Truncating %d values by %d",
                         len(values11), truncates)
            values11 = truncate(values11, truncates)
            values21 = truncate(values21, truncates)

        logger.debug("Averaging %d values", len(values11))
        values11 = np.average(values11, 0).tolist()
        values21 = np.average(values21, 0).tolist()

        return freq, values11, values21
    
    def readSegment(self, start, stop):
        logger.debug("Setting sweep range to %d to %d", start, stop)
        self.vna.setSweep(start, stop)

        frequencies = self.vna.readFrequencies()
        logger.debug("Read %s frequencies", len(frequencies))
        values11 = self.readData("data 0")
        values21 = self.readData("data 1")
        if not len(frequencies) == len(values11) == len(values21):
            logger.info("No valid data during this run")
            return [], [], []
        return frequencies, values11, values21

    def readData(self, data):
        logger.debug("Reading %s", data)
        done = False
        returndata = []
        count = 0
        while not done:
            done = True
            returndata = []
            tmpdata = self.vna.readValues(data)
            logger.debug("Read %d values", len(tmpdata))
            for d in tmpdata:
                a, b = d.split(" ")
                try:
                    if self.vna.validateInput and (
                            abs(float(a)) > 9.5 or
                            abs(float(b)) > 9.5):
                        logger.warning(
                            "Got a non plausible data value: (%s)", d)
                        done = False
                        break
                    returndata.append((float(a), float(b)))
                except ValueError as exc:
                    logger.exception("An exception occurred reading %s: %s",
                                     data, exc)
                    done = False
            if not done:
                logger.debug("Re-reading %s", data)
                sleep(0.2)
                count += 1
                if count == 5:
                    logger.error("Tried and failed to read %s %d times.",
                                 data, count)
                    logger.debug("trying to reconnect")
                    self.vna.reconnect()
                if count >= 10:
                    logger.critical(
                        "Tried and failed to read %s %d times. Giving up.",
                        data, count)
                    raise IOError(
                        f"Failed reading {data} {count} times.\n"
                        f"Data outside expected valid ranges,"
                        f" or in an unexpected format.\n\n"
                        f"You can disable data validation on the"
                        f"device settings screen.")
        return returndata

    def set_calibration(self, calibration_file:str):
        self.calibration.load(calibration_file)
        self.calibration.useIdealShort = True
        self.calibration.useIdealOpen = True
        self.calibration.useIdealLoad = True
        self.calibration.useIdealThrough = True
        self.calibration.calc_corrections()

    def apply_calibration(self,
                         raw_data11: List[Datapoint],
                         raw_data21: List[Datapoint]
                         ) -> Tuple[List[Datapoint], List[Datapoint]]:
        '''if self.offsetDelay != 0:
            tmp = []
            for dp in raw_data11:
                tmp.append(correct_delay(dp, self.offsetDelay, reflect=True))
            raw_data11 = tmp
            tmp = []
            for dp in raw_data21:
                tmp.append(correct_delay(dp, self.offsetDelay))
            raw_data21 = tmp
        '''
        if not self.calibration.isCalculated:
            logger.debug("calibration not calculated")
            return raw_data11, raw_data21

        data11: List[Datapoint] = []
        data21: List[Datapoint] = []

        if self.calibration.isValid1Port():
            logger.debug("correcting s11")
            for dp in raw_data11:
                data11.append(self.calibration.correct11(dp))
        else:
            logger.debug("s11 calibration not valid, handing raw data")
            data11 = raw_data11

        if self.calibration.isValid2Port():
            logger.debug("correcting s21")
            for dp in raw_data21:
                data21.append(self.calibration.correct21(dp))
        else:
            logger.debug("s21 calibration not valid, handing raw data")
            data21 = raw_data21
        return data11, data21

    def exportFile(self, filename:str, data: List[List[Datapoint]]):

        ts = Touchstone(filename)
        i = 0


        
        ts.sdata[0] = data[0]
        ts.sdata[1] = data[1]
        for dp in data[0]:
            ts.sdata[2].append(Datapoint(dp.freq, 0, 0))
            ts.sdata[3].append(Datapoint(dp.freq, 0, 0))
        try:
            ts.save(4)
        except IOError as e:
            logger.exception("Error during file export: %s", e)
            return



def plot_polar(complex_parameter, parameter_name = ""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')

    r = np.abs(complex_parameter)
    theta = np.angle(complex_parameter)
    ax.scatter(theta, r, s=3)

    ax.set_rmax(1)
    #ax.set_rticks([0.25, 0.5, 0.75, 1])  # less radial ticks
    #ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(True)

    ax.set_title("parameter "+parameter_name, va='bottom')

def plot_magnitude(freq, complex_parameter, parameter_name = ""):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    g = 10*np.log10(np.abs(complex_parameter))
    
    print("e len fregg", len(freq))
    print("eee len ggggg", len(g))

    print("freqqq", freq)
    print("ggg", g)

    ax.scatter(freq, g, s=3)


    ax.grid(True)

    ax.set_title("parameter "+parameter_name, va='bottom')

def convert_2d_float_1d_complex(_2d_float):
    _1d_complex = []
    for number in _2d_float:
        _1d_complex.append(complex(number[0], number[1]))

    return _1d_complex

if __name__ == "__main__":
  main()

def main():
    my_test = MyTest()
    logger.info("loading calibration")
    my_test.set_calibration("calibrations/24_25_includedKit_blueCable.cal")
    logger.info("connecting to device")
    my_test.connect_device()



    logger.info("read averaged sweep")
    
    averaged_read = my_test.readAveragedSegment(2400000, 2500000, 3, 0)
    logger.debug("\nthe frequencies("+str(len(averaged_read[0]))+"): "+str(averaged_read[0]))
    logger.debug("\ns11 data("+str(len(averaged_read[1]))+"): "+str(averaged_read[1]))
    logger.debug("\ns21 data("+str(len(averaged_read[2]))+"): "+str(averaged_read[2]))

    logger.info("applying calibration")
    #build data points out of the read
    s11_raw = []
    s21_raw = []
    for i in range(len(averaged_read[0])):
        s11_dp = Datapoint( freq = averaged_read[0][i],\
                            re = averaged_read[1][i][0],\
                             im = averaged_read[1][i][1]   )
        s11_raw.append(s11_dp)
        s21_dp = Datapoint( freq = averaged_read[0][i],\
                            re = averaged_read[2][i][0],\
                             im = averaged_read[2][i][1]   )
        s21_raw.append(s21_dp)
        
    def printdatapoints(datapoints:List[Datapoint]):
        for dp in datapoints:
            print(dp)

    print("s11_raw:")
    printdatapoints(s11_raw)
    print("s21_raw:")
    printdatapoints(s21_raw)
        
    my_test.exportFile("/home/stone/GIT/nanovna-saver/touchstoneTest.csv", [s11_raw, s21_raw])

    calibrated_read__ = my_test.apply_calibration(s11_raw, s21_raw)

    calibrated_read = []
    for i in range(3):
        calibrated_read.append([])
    for i in range(len(calibrated_read__[0])):
        calibrated_read[0].append(calibrated_read__[0][i].freq)
        calibrated_read[1].append([calibrated_read__[0][i].re, calibrated_read__[0][i].im])
        calibrated_read[2].append([calibrated_read__[1][i].re, calibrated_read__[1][i].im])


    logger.debug("\nthe calibrated read: "+str(calibrated_read))
    logger.info("displaying data")
    plot_magnitude(averaged_read[0], convert_2d_float_1d_complex(averaged_read[1]), "uncalibrated s11")
    plot_magnitude(averaged_read[0], convert_2d_float_1d_complex(averaged_read[2]), "uncalibrated s21")
    plot_magnitude(calibrated_read[0], convert_2d_float_1d_complex(calibrated_read[1]), "calibrated s11")
    plot_magnitude(calibrated_read[0], convert_2d_float_1d_complex(calibrated_read[2]), "calibrated s21")
    plt.show()

    #save touchstone files for further data processing and visualization

    logger.info("disconnecting device")

    my_test.disconnect_device()
