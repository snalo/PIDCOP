
#!/usr/bin/env python3
import sys
import math
import json
import pandas as pd

# Load model configurations
with open("model_configurations.json", "r") as f:
    model_configs = json.load(f)

# Load hardware configurations
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

# Speed of light and microring parameters (fixed)
s_light = 299792458
MR_radius = 5e-6
MR_seperation = 5e-6

results = []

for model_name, hw_name in combinations:
    config = model_configs[model_name]
    hw = hardware_configs[hw_name]

    # Model parameters
    H_model = config["H_model"]
    N_layers = config["N_layers"]
    d_model = config["d_model"]
    dff = config["dff"]
    d_sequence = config["d_sequence"]
    num_tokens = config["num_tokens"]

    # Hardware parameters
    mul_lanes = hw["mul_lanes"]
    mul_cols = hw["mul_cols"]
    num_attn_units = hw["num_attn_units"]
    num_ffn_units = hw["num_ffn_units"]
    bit_rate = hw["bit_rate"]
    stochastic_bits = hw["stochastic_bits"]

    bits_gen_T = stochastic_bits / bit_rate

    # STEP 1
    dist_matmul = (3 * MR_radius * mul_cols) * MR_radius + ((3 * MR_radius * mul_cols) - 1) * MR_seperation
    T_matmul_p1 = dist_matmul / s_light
    num_muls_1 = num_tokens * d_model * d_model * 3 * N_layers
    num_muls_per_unit_1 = math.ceil(num_muls_1 / (mul_lanes * mul_cols * num_attn_units))
    muls_T_1 = num_muls_per_unit_1 * bits_gen_T

    # STEP 2
    num_muls_2 = num_tokens * num_tokens * d_model * N_layers
    num_muls_per_unit_2 = math.ceil(num_muls_2 / (mul_lanes * mul_cols * num_attn_units))
    muls_T_2 = num_muls_per_unit_2 * bits_gen_T

    # STEP 3
    softmax_T = (623.7 + 719.95 + 222.5 + 719.95 + 222.5) * 1e-12

    # STEP 4
    num_muls_4 = num_tokens * num_tokens * d_model * N_layers
    num_muls_per_unit_4 = math.ceil(num_muls_4 / (mul_lanes * mul_cols * num_attn_units))
    muls_T_4 = num_muls_per_unit_4 * bits_gen_T

    # STEP 5
    num_muls_5 = num_tokens * d_model * dff * N_layers
    num_muls_per_unit_5 = math.ceil(num_muls_5 / (mul_lanes * mul_cols * num_ffn_units))
    muls_T_5 = num_muls_per_unit_5 * bits_gen_T

    # STEP 6
    num_muls_6 = num_tokens * d_model * dff * N_layers
    num_muls_per_unit_6 = math.ceil(num_muls_6 / (mul_lanes * mul_cols * num_ffn_units))
    muls_T_6 = num_muls_per_unit_6 * bits_gen_T

    # Total Latency
    total_time_encoder_mha = muls_T_1 + muls_T_2 + softmax_T + muls_T_4
    total_time_encoder_ffn = muls_T_5 + muls_T_6
    total_time = total_time_encoder_mha + total_time_encoder_ffn

    results.append({
        "Model": model_name,
        "Hardware": hw_name,
        "N_layers": N_layers,
        "d_model": d_model,
        "dff": dff,
        "H_model": H_model,
        "Tokens": num_tokens,
        "Latency_seconds": total_time
    })

# Save to Excel
df = pd.DataFrame(results)
df.to_excel("results/latency_sim_model_hw_results.xlsx", index=False, engine="openpyxl")

print("Simulation complete. Results saved to 'latency_sim_model_hw_results.xlsx'")
