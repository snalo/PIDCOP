
#!/usr/bin/env python3
import sys
import math
import json
import pandas as pd

# Load model configuration
with open("model_configurations.json", "r") as f:
    model_configs = json.load(f)

# Load hardware configuration
with open("hardware_configurations.json", "r") as f:
    hardware_configs = json.load(f)

'''
# Define combinations of (model, hardware) to simulate
combinations = [
    ("BERT-base", "ASTRA_HW"),
    ("GPT-2 Medium", "ASTRA_HW"),
    ("T5-base", "Archi_1"),
    ("ViT Base", "Archi_2"),
    ("DeiT-Small", "ASTRA_HW")
]
# Generate all combinations: each model with every hardware config
model_list = ["BERT-base", "GPT-2 Medium", "T5-base", "ViT Base", "DeiT-Small"]
hardware_list = ["ASTRA_HW", "Archi_1", "Archi_2"]
'''
model_list = list(model_configs.keys())
hardware_list = list(hardware_configs.keys())

combinations = [(model, hw) for model in model_list for hw in hardware_list]

results = []

for model_name, hw_name in combinations:
    config = model_configs[model_name]
    hw = hardware_configs[hw_name]

    # -------------------- Model Parameters -------------------- #
    H_model = config["H_model"]
    N_layers = config["N_layers"]
    d_model = config["d_model"]
    dff = config["dff"]
    d_sequence = config["d_sequence"]
    num_tokens = config["num_tokens"]
    DECODER = 0

    # -------------------- Architecture Parameters -------------------- #
    mul_lanes = hw["mul_lanes"]
    mul_cols = hw["mul_cols"]
    num_units = hw["num_attn_units"]
    num_ffn_units = hw["num_ffn_units"]
    bit_rate = hw["bit_rate"]
    stochastic_bits = hw["stochastic_bits"]

    # -------------------- Components -------------------- #
    s_light = 299792458
    PDs_T = 5.8e-12
    ADC_T = 0.78e-9
    DAC_T = 0
    VCSEL_T = 10e-9
    SOA_L = 10e-6
    B_to_S_T = 530e-12
    BPCA_T = 2

    MR_radius = 5e-6
    MR_seperation = 5e-6

    ADC_power = 2e-3
    EO_power = 1.6e-6
    TO_power = 1.375e-3
    PCA_power = 0.02e-3
    SER_power = 0.0015
    SOA_power = 2.2e-3
    B_to_TCU_power = 0.021e-3
    digital_ACT_power = 0.52e-3
    softmax_power = (0.055 + 0.0014 + 4.21) * 1e-3

    signal_prop = 1
    split_loss = 0.13
    MR_thru_loss = 0.02
    MR_mod_loss = 0
    laser_power_per_wl = 1.274274986e-3

    bits_gen_T = stochastic_bits / bit_rate

    # ADC Energy
    energy_adc = ADC_power * ADC_T
    num_ADC = num_units * mul_lanes
    adc_energy_total = energy_adc * num_ADC

    # Non-linear Energy
    num_non_linear = num_units
    power_non_linear = (4.2664 + 4.21) * 1e-3
    latency_non_linear = (1.89 + 0.2225) * 1e-9
    energy_non_linear = power_non_linear * latency_non_linear
    non_linear_energy_total = energy_non_linear * num_non_linear

    # Serializers
    num_ser_mul_unit_M1 = mul_cols * num_units
    num_ser_mul_unit_M2 = mul_cols * mul_lanes
    total_num_ser_2 = num_ser_mul_unit_M1 + num_ser_mul_unit_M2
    energy_ser = SER_power * 0.03e-9
    ser_energy_total = energy_ser * total_num_ser_2

    # OAG
    num_oag = num_units * mul_cols * mul_lanes
    energy_oag = TO_power * 3.33333e-11
    oag_energy_total = energy_oag * num_oag

    # B to S
    num_b_to_s = total_num_ser_2
    energy_b_to_s = B_to_TCU_power * B_to_S_T
    b_to_s_energy_total = energy_b_to_s * num_b_to_s

    # PCA
    num_pca = num_units * mul_lanes * 2
    energy_pca = PCA_power * 2.19058e-9
    pca_total_energy = energy_pca * num_pca

    # Comb Laser
    num_laser = num_units
    energy_laser = 0.075439453125 * 3.33333e-11
    laser_total_energy = energy_laser * num_laser

    total_energy = adc_energy_total + non_linear_energy_total + ser_energy_total + oag_energy_total + b_to_s_energy_total + pca_total_energy + laser_total_energy

    results.append({
        "Model": model_name,
        "Hardware": hw_name,
        "ADC Energy": adc_energy_total,
        "Nonlinear Energy": non_linear_energy_total,
        "Serializer Energy": ser_energy_total,
        "OAG Energy": oag_energy_total,
        "B_to_S Energy": b_to_s_energy_total,
        "PCA Energy": pca_total_energy,
        "Laser Energy": laser_total_energy,
        "Total Energy": total_energy
    })

# Save to Excel
df = pd.DataFrame(results)
df.to_excel("results/energy_breakdown_results.xlsx", index=False, engine="openpyxl")
print("Batch energy simulation complete. Results saved to 'energy_breakdown_results.xlsx'")
