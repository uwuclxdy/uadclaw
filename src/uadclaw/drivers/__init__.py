"""One module per OEM, each a single `FirmwareDriver` implementation.

A package rather than a module because tasks 11's six further OEMs (Xiaomi, Nothing,
Motorola, Samsung, Oppo/OnePlus/Realme) each land here as one more file, and each must be
isolated enough to be disabled in configuration without any other stage noticing.
"""
