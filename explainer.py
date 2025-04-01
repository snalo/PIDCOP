'''
Model Parameters
These define the architecture:
H_model: Number of attention heads.
N_layers: Number of transformer layers.
d_model: Embedding dimension (size of input/output vectors).
dff: Size of feed-forward hidden layer.
d_sequence: Sequence length.
num_tokens: Number of tokens in a batch.
DECODER = 0: Indicates this is an encoder-only architecture.

Architecture Parameters
Describes the hardware:
mul_lanes, mul_cols: Parallel processing units for multiplication.
Divide the multiplication to rows and columns - M - parallelism
num_attn_units, num_ffn_units: Units allocated for attention and FFN (feed-forward network).
bit_rate: Transmission rate (30 Gbps).
stochastic_bits: Bitwidth for computations (8 bits + sign bit).

Component Latencies
Describes delays for various optical/electronic components:
E.g., ADC_T, DAC_T, VCSEL_T, SOA_L etc.
s_light: Speed of light used for optical delay calculations.
bits_gen_T: Time to generate 129 bits from bitstream.
'''
