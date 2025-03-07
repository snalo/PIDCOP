"""
This Python script calculates the total area of an integrated photonic Mach-Zehnder Interferometer (MZI).
The script is designed to be modular, allowing you to easily change the dimensions of the MZI components
based on your research or design requirements.

The MZI consists of:
1. Input Waveguide
2. Y-Junction or Directional Coupler (Splitter)
3. Two Interferometer Arms
4. Y-Junction or Directional Coupler (Combiner)
5. Output Waveguide
6. Optional Features (e.g., phase shifters, heaters)

The script calculates the area of each component and sums them up to provide the total area.
"""

import math

waveguide_width = 0.05  # Width of the waveguide (µm)
arm_length = 500  # Length of each interferometer arm (µm)
bend_radius = 5  # Radius of curved sections (µm)
spacing_between_arms = 2  # Distance between the two arms (µm)
y_junction_length = 5  # Length of the Y-junction or coupler (µm)
y_junction_width = 1  # Width of the Y-junction or coupler (µm)

# Calculate the area of each component

area_straight_waveguides = 2 * (arm_length * waveguide_width) # Straight Waveguides (two arms)
area_curved_waveguides = 2 * (math.pi * bend_radius**2) # Curved Waveguides (two bends)
area_y_junctions = 2 * (y_junction_length * y_junction_width) # Y-Junctions (two Y-junctions)
total_area = area_straight_waveguides + area_curved_waveguides + area_y_junctions # Total Area (sum of all components)

# Compact Layout Area (approximation for a compact design)
# This formula accounts for the spacing between arms and the waveguide width
compact_area = (arm_length + 2 * bend_radius) * (spacing_between_arms + 2 * waveguide_width)

# Print the results
print("MZI Area Calculation Results:")
print(f"1. Area of Straight Waveguides: {area_straight_waveguides:.2f} µm²")
print(f"2. Area of Curved Waveguides: {area_curved_waveguides:.2f} µm²")
print(f"3. Area of Y-Junctions: {area_y_junctions:.2f} µm²")
print(f"4. Total Area (Sum of Components): {total_area:.2f} µm²")
print(f"5. Compact Layout Area: {compact_area:.2f} µm²")

# Convert total area to mm² for easier interpretation
total_area_mm2 = total_area / 1e6
compact_area_mm2 = compact_area / 1e6
print(f"\nTotal Area in mm²: {total_area_mm2:.6f} mm²")
print(f"Compact Layout Area in mm²: {compact_area_mm2:.6f} mm²")

#https://www.engr.colostate.edu/~mnikdast/files/papers/Mahdi_J32.pdf