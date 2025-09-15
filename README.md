# battery-pack
Li-Ion 21700 12S3P Pack

## notes
Battery will have onboard 12V buck regulator for aux power (5A-10A)
This will also have double gated 12V to external aux power

12V -> 5V buck regulator for mini aux power (VCC self power effeciency and for BMS util)

BMS takes in 5V and LDO to 3V3

Main BMS: read all cell voltages, balancing, current measure, and dual gate disconnect, temperature
put one thermistor on the board and some on the cells

On the back of the BMS and maybe the regulator, the thermistor connections to measure cell temps
