# from PerformanceMetrics.metrics import Metrics
#
# metrics = Metrics()
# AREA = "area"
# POWER = "power"
# #Area in mm2  and power in mW
# adc_area_power = {
#     1: {AREA: 2, POWER: 2.55},
#     5: {AREA: 21, POWER:11},
#     10: {AREA: 103, POWER: 14.8}
# }
# #Area in mm2 and power in mW
# dac_area_power = {
#     1: {AREA: 0.00007, POWER: 0.12},
#     5: {AREA: 0.06, POWER: 26},
#     10: {AREA: 0.06, POWER: 30}
# }
# # ANALOG_ACCELERATOR = [{ELEMENT_SIZE: 4, ELEMENT_COUNT: 33, UNITS_COUNT: 200, RECONFIG: [
# # ], VDP_TYPE:'AMM', NAME:'DEAPCNN', ACC_TYPE:'ANALOG', PRECISION:4, BITRATE: 1}]
# # ANALOG_ACCELERATOR = [{ELEMENT_SIZE: 36, ELEMENT_COUNT: 36, UNITS_COUNT: 200, RECONFIG: [
# # ], VDP_TYPE:'AMM', NAME:'HOLYLIGHT', ACC_TYPE:'ANALOG', PRECISION:4, BITRATE: 1}]
# # ANALOG_ACCELERATOR = [{ELEMENT_SIZE: 43, ELEMENT_COUNT: 43, UNITS_COUNT: 200, RECONFIG: [
# # ], VDP_TYPE:'AMM', NAME:'SPOGA_10', ACC_TYPE:'ANALOG', PRECISION:4, BITRATE: 1}]
# # area = metrics.get_total_area(vdp_type, accelearator_config[UNITS_COUNT], 0, accelearator_config[ELEMENT_SIZE],
# #                                accelearator_config[ELEMENT_COUNT], 0, 0, accelearator_config[RECONFIG],
# #                                accelearator_config[ACC_TYPE])
# running_br = 1
# metrics.dac.area = dac_area_power[running_br][AREA]
# metrics.adc.area = adc_area_power[running_br][AREA]
# #UNIT COUNTS = 150, ELEMENT SIZE = 249, ELEMENT COUNT = 16,
#
# area = metrics.get_total_area('PIDCOP',250, 193, 8)
# print("Area_per_unit_PIDCOP", area)
#
# area = metrics.get_total_area('DEAPCNN', 135, 15, 15)
# print("Area_per_unit_DEAPCNN", area)
#
# area = metrics.get_total_area('HOLYLIGHT',50,43,43)
# print("Area_per_unit_HOLYLIGHT", area)
#
# area = metrics.get_total_area('SPOGA',2, 249, 16) #Target Area = 15716 FOR pidcop
# print("Area_per_unit_SPOGA", area)
#
# area = metrics.get_total_area('PIXEL_OO',500, 4, 4)
# print("Area_per_unit_PIXEL_OO", area)
#
# area = metrics.get_total_area('PIXEL_EO',500, 4, 4)
# print("Area_per_unit_PIXEL_EO", area)

from PerformanceMetrics.metrics import Metrics

metrics = Metrics()
AREA = "area"
POWER = "power"

# Area in mm^2 and power in mW for ADC and DAC
adc_area_power = {
    1: {AREA: 2, POWER: 2.55},
    5: {AREA: 21, POWER: 11},
    10: {AREA: 103, POWER: 14.8}
}

dac_area_power = {
    1: {AREA: 0.00007, POWER: 0.12},
    5: {AREA: 0.06, POWER: 26},
    10: {AREA: 0.06, POWER: 30}
}

# Accelerator configurations for different running_br values
accelerator_configs = {
    1: [
        ('PIDCOP', 250, 180, 8),
        ('DEAPCNN', 135, 15, 15),
        ('HOLYLIGHT', 50, 43, 43),
        ('SPOGA', 2, 249, 16),
        ('PIXEL_OO', 500, 4, 4),
        ('PIXEL_EO', 500, 4, 4)
    ],
    5: [
        ('PIDCOP', 25, 118, 85),
        ('DEAPCNN', 13, 15, 155),
        ('HOLYLIGHT', 40, 436, 43),
        ('SPOGA', 2, 24, 166),
        ('PIXEL_OO', 550, 45, 42),
        ('PIXEL_EO', 510, 34, 54)
    ],
    10: [
        ('PIDCOP', 243, 91, 85),
        ('DEAPCNN', 163, 15, 155),
        ('HOLYLIGHT', 410, 436, 43),
        ('SPOGA', 2, 24, 16566),
        ('PIXEL_OO', 550, 465, 42),
        ('PIXEL_EO', 510, 3421, 54)
    ]
}

# Iterate over all running_br values (1, 5, 10)
for running_br in [1, 5, 10]:
    print(f"\n=== Running for running_br = {running_br} ===")

    # Assign DAC and ADC area values based on running_br
    metrics.dac.area = dac_area_power[running_br][AREA]
    metrics.adc.area = adc_area_power[running_br][AREA]

    # Select corresponding accelerator configurations
    configs = accelerator_configs[running_br]

    # Compute and print areas for the selected accelerator configurations
    for vdp_type, unit_count, element_size, element_count in configs:
        area = metrics.get_total_area(vdp_type, unit_count, element_size, element_count)
        print(f"Area_per_unit_{vdp_type} (running_br={running_br}): {area}")



