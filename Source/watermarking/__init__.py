"""
SecureDICOM Watermarking Module

Robust FFT-based digital watermarking for medical images.
"""

from .robust_watermark import (
    embed_robust_watermark,
    extract_robust_watermark,
    payload_to_bits,
    bits_to_payload,
    WatermarkException,
)

from .watermark_patterns import (
    LFSR,
    generate_bit_sequence,
    generate_patient_bit_sequence,
    create_ownership_payload,
    hash_to_seed,
)

from .fragile_watermark import (
    FragileConfig,
    FragileWatermarkException,
    embed_fragile_watermark,
    detect_tamper_map,
    tamper_stats,
)

from .tamper_detection import (
    localize_tamper_regions,
)

__all__ = [
    'embed_robust_watermark',
    'extract_robust_watermark',
    'payload_to_bits',
    'bits_to_payload',
    'WatermarkException',
    'LFSR',
    'generate_bit_sequence',
    'generate_patient_bit_sequence',
    'create_ownership_payload',
    'hash_to_seed',
    'FragileConfig',
    'FragileWatermarkException',
    'embed_fragile_watermark',
    'detect_tamper_map',
    'tamper_stats',
    'localize_tamper_regions',
]

__version__ = "1.0.0"
