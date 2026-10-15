# Linux Six-Sensor Monitor

This directory contains the Linux version of the six-channel gas-sensor monitor.
It receives data from a serial device, plots all six sensor values in real time,
and saves recorded samples as a CSV file.

## Data protocol

The device must send one UTF-8 text frame per line. Each frame has exactly eight
comma-separated fields, in this order:

```text
sensor1,sensor2,sensor3,sensor4,sensor5,sensor6,alarm_status,alarm_channel
```

Example:

```text
1.20,1.35,0.98,2.10,1.67,1.44,0,0
```

`alarm_status` is `0` for normal and nonzero for alarm. `alarm_channel` is `0`
when there is no alarm, otherwise it is the alarm channel number.

## First-time setup

The following commands are for Ubuntu/Debian. Use the equivalent package command
on another distribution.

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-tk
cd linux_sensor_monitor
chmod +x run_sensor_monitor.sh
./run_sensor_monitor.sh
```

The launcher creates an isolated `.venv` directory and installs `pyserial` and
`matplotlib` automatically. Internet access is required only for this initial
package installation and later dependency upgrades.

## Serial-port permission

After connecting the USB serial device, determine its device file:

```bash
ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
```

If the program cannot open that port, add the current user to the `dialout`
group, then log out and log in again:

```bash
sudo usermod -aG dialout $USER
```

In the application, select the displayed `/dev/ttyUSB0`, `/dev/ttyACM0`, or
other serial-device entry, then click **Connect serial port**. The baud rate is
fixed at 9600 in `sensor_monitor.py`; change `BAUD_RATE` near the top of that
file if the firmware uses a different baud rate.

## Recording

1. Click **Start recording** after connecting.
2. Click **Stop and save** when collection is complete.
3. Choose a filename in the save dialog. The CSV has timestamp, six sensor
   columns, alarm status, and alarm channel columns. It is saved as UTF-8 with
   BOM so it opens directly in spreadsheet applications.
