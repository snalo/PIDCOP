
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

    # Model
    H = config["H_model"]
    N_layers = config["N_layers"]
    d_model = config["d_model"]
    d_sequence = config["d_sequence"]

    # Hardware
    mul_lanes = hw["mul_lanes"]
    mul_cols = hw["mul_cols"]
    num_attn_units = hw["num_attn_units"]
    num_ffn_units = hw["num_ffn_units"]
    bit_rate = hw["bit_rate"]
    stochastic_bits = hw["stochastic_bits"]

    # Component Power (W)
    ADC_power = 2e-3
    DAC_power = 0.0078e-3
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

    MR_radius = 5
    MR_seperation = 5

    # Comb Laser Power
    comb_laser_p = 0.5
    comb_laser_p_total = comb_laser_p * (num_attn_units + num_ffn_units)

    # ADC Power
    num_adc_attns = mul_lanes * 2 * num_attn_units
    num_adc_ffns = 1 * num_ffn_units
    total_power_adc_1 = (num_adc_attns + num_adc_ffns) * ADC_power

    # PCA Power
    total_num_pca = mul_lanes * 2 * (num_attn_units + num_ffn_units)
    total_power_pca = total_num_pca * PCA_power

    # Serializer Power Option 1
    total_num_ser_1 = mul_lanes * mul_cols * 2 * (num_attn_units + num_ffn_units)
    total_power_ser_1 = total_num_ser_1 * SER_power

    # Serializer Power Option 2
    ser_shared_units = mul_lanes * mul_cols
    total_num_ser_2 = (ser_shared_units * num_attn_units + ser_shared_units +
                       ser_shared_units * num_ffn_units + ser_shared_units)
    total_power_ser_2 = total_num_ser_2 * SER_power

    # MR Tuning
    total_num_oags = mul_lanes * mul_cols * (num_attn_units + num_ffn_units)
    total_power_oags_to = total_num_oags * TO_power
    total_num_fmrs = mul_lanes * mul_cols * 2 * (num_attn_units + num_ffn_units)
    total_power_fmrs_eo = total_num_fmrs * EO_power

    # Softmax + Activation
    total_power_softmax = num_attn_units * softmax_power
    total_power_soa = num_ffn_units * SOA_power
    total_power_act = num_ffn_units * digital_ACT_power

    # B_to_S Power Option 1
    total_num_b_to_s_1 = mul_lanes * mul_cols * 2 * (num_attn_units + num_ffn_units)
    total_power_b_to_s_1 = total_num_b_to_s_1 * B_to_TCU_power

    # B_to_S Power Option 2
    b2s_shared_units = mul_lanes * mul_cols
    total_num_b_to_s_2 = (b2s_shared_units * num_attn_units + b2s_shared_units +
                          b2s_shared_units * num_ffn_units + b2s_shared_units)
    total_power_b_to_s_2 = total_num_b_to_s_2 * B_to_TCU_power

    total_power_optical_1 = (comb_laser_p_total + total_power_adc_1 + total_power_pca +
                             total_power_ser_1 + total_power_oags_to + total_power_fmrs_eo +
                             total_power_softmax + total_power_soa + total_power_b_to_s_1)

    total_power_optical_2 = (comb_laser_p_total + total_power_adc_1 + total_power_pca +
                             total_power_ser_2 + total_power_oags_to + total_power_fmrs_eo +
                             total_power_softmax + total_power_soa + total_power_b_to_s_2)

    results.append({
        "Model": model_name,
        "Hardware": hw_name,
        "Power_no_OS": total_power_optical_1,
        "Power_with_OS": total_power_optical_2,
        "Laser Power": comb_laser_p_total
    })

# Save to Excel
df = pd.DataFrame(results)
df.to_excel("results/power_sim_results.xlsx", index=False, engine="openpyxl")
print("Batch power simulation complete. Results saved to 'power_sim_results.xlsx'")
