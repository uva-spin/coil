import pyvisa

SMA_IP_ADDRESS = "192.168.1.10"
SMA_ETHERNET_ADDRESS = f"TCPIP0::{SMA_IP_ADDRESS}::inst0::INSTR"

target_freq_mhz = 213.0
target_power_dbm = -10.0

rm = pyvisa.ResourceManager()
sma = rm.open_resource(SMA_ETHERNET_ADDRESS, timeout=5000)
print(f"Connected to RF Gen: {sma.query('*IDN?').strip()}")

try:
    sma.write("*RST")

    target_freq_hz = target_freq_mhz * 1e6
    print(f"\nSetting single frequency bin to {target_freq_mhz} MHz at {target_power_dbm} dBm...")
    sma.write(f"SOURce:FREQuency:CW {target_freq_hz}")
    sma.write(f"SOURce:POWer:LEVel {target_power_dbm}")
    sma.write("OUTPut:STATe ON")

    current_freq = float(sma.query("SOURce:FREQuency:CW?"))
    current_pow = float(sma.query("SOURce:POWer:LEVel?"))
    print("\nVerification from Hardware:")
    print(f"-> Active Frequency: {current_freq / 1e6:.2f} MHz")
    print(f"-> Active Power Level: {current_pow:.2f} dBm")

    input(f"\nRF Output is LIVE at {target_freq_mhz} MHz @ {target_power_dbm} dBm. "
          "Press Enter to safely turn off output and close...")
finally:
    sma.write("OUTPut:STATe OFF")
    sma.close()
    print("Generator safely disconnected.")
