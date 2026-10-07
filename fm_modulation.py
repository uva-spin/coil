#!/usr/bin/env python3
import sys
import pyvisa

SMA_IP_ADDRESS = "192.168.1.10"
SMA_ADDRESS = f"TCPIP0::{SMA_IP_ADDRESS}::inst0::INSTR"

CARRIER_FREQ_HZ = 213e6      # f0
POWER_LEVEL_DBM = 1.0        

FM_SPAN_HZ = 400e3           
FM_DEVIATION_HZ = FM_SPAN_HZ / 2  # SOURce:FM:DEViation is one-sided (peak), so span = 2x deviation

MOD_RATE_HZ = 1.0            # LF generator (triangle) rate, in Hz - GUESS from VI's T_sweep=1s

FM_WAVEFORM = "TRIangle"     # SINE | SQUare | PULSe | TRIangle | TRAPeze
FM_MODE = "HBANdwidth"       # HBANdwidth (max mod bandwidth) | LNOise


def main():
    rm = pyvisa.ResourceManager("@py")
    sma = rm.open_resource(SMA_ADDRESS, timeout=5000)
    print(f"Connected to RF Gen: {sma.query('*IDN?').strip()}")

    sma.write("*RST")
    sma.write("*CLS")

    # Carrier frequency and power
    sma.write(f"SOURce1:FREQuency:CW {CARRIER_FREQ_HZ}")
    sma.write(f"SOURce1:POWer:LEVel:IMMediate:AMPLitude {POWER_LEVEL_DBM}")

    # Internal LF generator: waveform + rate
    sma.write(f"SOURce1:LFOutput1:SHAPe {FM_WAVEFORM}")
    sma.write(f"SOURce1:LFOutput1:FREQuency {MOD_RATE_HZ}")

    # FM path 1: use the internal LF generator (LF1) as the modulation source
    sma.write("SOURce1:FM1:SOURce LF1")
    sma.write(f"SOURce1:FM1:DEViation {FM_DEVIATION_HZ}")
    sma.write(f"SOURce1:FM:MODe {FM_MODE}")

    # Turn on FM, the LF generator, and RF output - equivalent of toggling
    # Path 1's "State" to 1 and RF to "On" on the touchscreen
    sma.write("SOURce1:FM1:STATe 1")
    sma.write("SOURce1:LFOutput1:STATe 1")
    sma.write("OUTPut1:STATe 1")

    fm_state = sma.query("SOURce1:FM1:STATe?").strip()
    rf_state = sma.query("OUTPut1:STATe?").strip()
    print(f"FM1 state: {fm_state}, RF output state: {rf_state}")
    print(f"Carrier: {CARRIER_FREQ_HZ/1e6:.3f} MHz, Deviation: +/-{FM_DEVIATION_HZ/1e3:.1f} kHz, "
          f"Rate: {MOD_RATE_HZ/1e3:.1f} kHz, Waveform: {FM_WAVEFORM}")

    sma.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--off":
        rm = pyvisa.ResourceManager("@py")
        sma = rm.open_resource(SMA_ADDRESS, timeout=5000)
        sma.write("SOURce1:FM1:STATe 0")
        sma.write("OUTPut1:STATe 0")
        print("FM and RF output turned off.")
        sma.close()
    else:
        main()
