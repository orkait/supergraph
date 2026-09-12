"""Pure algorithmic primitives.

Contract (enforced):
    - No imports from supergraph.* - only stdlib, numpy, scipy
    - No I/O, no logging, no global state
    - Inputs are numpy arrays / CSR matrices / plain data
    - Identical input → identical output
    - Caller owns side effects
"""
