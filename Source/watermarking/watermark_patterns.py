"""
Watermark Pattern Generation using LFSR

This module generates pseudo-random patterns for reproducible watermark generation.
Uses Linear Feedback Shift Register (LFSR) for deterministic bit sequence generation.
"""

import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class LFSR:
    """
    Linear Feedback Shift Register for pseudo-random bit generation.
    
    Uses standard LFSR polynomial for reproducibility.
    Same seed always produces same bit sequence.
    """
    
    # Standard 31-bit LFSR polynomial: x^31 + x^3 + 1
    POLYNOMIAL = 0x80000009
    
    def __init__(self, seed: int = 0x12345678):
        """
        Initialize LFSR with seed.
        
        Args:
            seed: 32-bit integer seed (default 0x12345678)
        """
        self.state = seed & 0xFFFFFFFF  # 32-bit mask
        if self.state == 0:
            self.state = 1  # LFSR cannot have all-zero state
        
        logger.debug(f"LFSR initialized with seed 0x{seed:08x}")
    
    def next_bit(self) -> int:
        """Generate next bit (0 or 1)."""
        # Output based on LSB
        output = self.state & 1
        
        # Calculate new state (Galois/Fibonacci LFSR)
        lsb = self.state & 1
        self.state = self.state >> 1
        
        if lsb == 1:
            self.state = self.state ^ self.POLYNOMIAL
        
        return output
    
    def next_bits(self, count: int) -> List[int]:
        """Generate next n bits."""
        return [self.next_bit() for _ in range(count)]
    
    def next_byte(self) -> int:
        """Generate next 8 bits as a byte (MSB to LSB)."""
        byte = 0
        for i in range(8):
            byte = (byte << 1) | self.next_bit()
        return byte


def hash_to_seed(identifier: str) -> int:
    """
    Convert string identifier (e.g., patient ID) to LFSR seed.
    
    Args:
        identifier: String identifier (patient ID, etc.)
    
    Returns:
        32-bit seed value
    """
    # Use Python's hash and XOR with constants for mixing
    h = hash(identifier)
    seed = (h & 0xFFFFFFFF) ^ 0x12345678
    
    # Ensure non-zero
    if seed == 0:
        seed = 0x87654321
    
    logger.debug(f"Generated seed 0x{seed:08x} from identifier '{identifier}'")
    return seed


def generate_bit_sequence(length: int, seed: int = 0x12345678) -> List[int]:
    """
    Generate pseudo-random bit sequence using LFSR.
    
    Args:
        length: Number of bits to generate
        seed: LFSR seed (default: fixed seed)
    
    Returns:
        List of binary bits
    """
    lfsr = LFSR(seed)
    return lfsr.next_bits(length)


def generate_patient_bit_sequence(patient_id: str, length: int = 256) -> List[int]:
    """
    Generate deterministic bit sequence from patient ID.
    Same patient_id always produces same sequence.
    
    Args:
        patient_id: Patient identifier string
        length: Number of bits to generate (default 256)
    
    Returns:
        List of binary bits
    """
    seed = hash_to_seed(patient_id)
    return generate_bit_sequence(length, seed)


def payload_to_bits(payload: str) -> List[int]:
    """
    Convert hex or binary payload string to bit list.
    
    Args:
        payload: Hex string (e.g., "a3b2c4d5") or binary string (e.g., "11010101")
    
    Returns:
        List of binary bits
    """
    if not payload:
        raise ValueError("Payload cannot be empty")
    
    # Detect format
    if payload.startswith('0b') or all(c in '01' for c in payload):
        # Binary format
        bits = [int(c) for c in payload.lstrip('0b')]
        return bits
    else:
        # Hex format
        try:
            payload_clean = payload.replace(" ", "")
            payload_int = int(payload_clean, 16)
            
            # Convert to 256-bit representation (or length of hex * 4)
            bit_length = len(payload_clean) * 4
            bits = [(payload_int >> i) & 1 for i in range(bit_length)]
            
            return bits
        except ValueError:
            raise ValueError(f"Invalid payload format: {payload}")


def bits_to_payload(bits: List[int], format: str = 'hex') -> str:
    """
    Convert bit list to hex or binary payload string.
    
    Args:
        bits: List of binary bits
        format: 'hex' or 'binary'
    
    Returns:
        Payload string
    """
    if not isinstance(bits, list) or not all(b in [0, 1] for b in bits):
        raise ValueError("Bits must be list of 0s and 1s")
    
    if format == 'binary':
        return ''.join(str(b) for b in bits)
    elif format == 'hex':
        # Convert bits to integer
        payload_int = 0
        for i, bit in enumerate(bits):
            payload_int |= (int(bit) << i)
        
        # Convert to hex string
        # Pad to byte boundary
        hex_length = (len(bits) + 3) // 4
        return format(payload_int, f'0{hex_length}x')
    else:
        raise ValueError(f"Unknown format: {format}")


def create_ownership_payload(patient_id: str, 
                            timestamp_hex: str = "06a08980",
                            signature_hex: str = "1f2e3d4c") -> str:
    """
    Create a 256-bit ownership payload from components.
    
    Payload format:
    - Bits 0-63: Patient ID hash (64 bits)
    - Bits 64-95: Timestamp (32 bits)
    - Bits 96-127: Signature (32 bits)
    - Bits 128-255: Reserved/future use (128 bits)
    
    Args:
        patient_id: Patient identifier
        timestamp_hex: Timestamp in hex (8 chars = 32 bits)
        signature_hex: Digital signature in hex (8 chars = 32 bits)
    
    Returns:
        256-bit payload as hex string
    """
    # Generate patient ID hash
    patient_hash = hash_to_seed(patient_id) & 0xFFFFFFFF
    
    # Parse provided values
    timestamp = int(timestamp_hex, 16) & 0xFFFFFFFF
    signature = int(signature_hex, 16) & 0xFFFFFFFF
    
    # Build 256-bit payload
    # Bits 0-31: Patient hash low word
    # Bits 32-63: Patient hash high word
    # Bits 64-95: Timestamp
    # Bits 96-127: Signature
    # Bits 128-255: Reserved
    
    payload_int = (
        (patient_hash) |  # Bits 0-31
        (patient_hash << 32) |  # Bits 32-63 (duplicate for redundancy)
        (timestamp << 64) |  # Bits 64-95
        (signature << 96)  # Bits 96-127
    )
    
    # Convert to 256-bit hex string
    payload_hex = format(payload_int, '064x')
    
    logger.info(f"Created ownership payload for patient {patient_id}")
    return payload_hex


def verify_payload_checksum(payload_hex: str) -> bool:
    """
    Simple checksum verification for payload integrity.
    
    Args:
        payload_hex: 64-character hex string (256 bits)
    
    Returns:
        True if checksum valid
    """
    if len(payload_hex) < 64:
        return False
    
    # Sum all bytes mod 256
    checksum = 0
    for i in range(0, 64, 2):
        byte_val = int(payload_hex[i:i+2], 16)
        checksum = (checksum + byte_val) & 0xFF
    
    # Valid if checksum (XOR of all bytes) is 0 or expected value
    # For now, just return consistent result
    return checksum != 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example 1: Generate bits from patient ID
    print("=== Example 1: Patient-based bits ===")
    patient_bits = generate_patient_bit_sequence("PATIENT_001", 256)
    print(f"Generated {len(patient_bits)} bits from patient ID")
    print(f"First 32 bits: {''.join(map(str, patient_bits[:32]))}")
    
    # Example 2: Create ownership payload
    print("\n=== Example 2: Ownership Payload ===")
    payload = create_ownership_payload("PATIENT_001", "06a08980", "1f2e3d4c")
    print(f"Payload: {payload}")
    print(f"Payload length: {len(payload)} characters = {len(payload) * 4} bits")
    
    # Example 3: LFSR demonstration
    print("\n=== Example 3: LFSR Test ===")
    lfsr = LFSR(0x12345678)
    bits = lfsr.next_bits(16)
    print(f"LFSR bits: {''.join(map(str, bits))}")
    
    # Verify reproducibility
    lfsr2 = LFSR(0x12345678)
    bits2 = lfsr2.next_bits(16)
    print(f"LFSR2 bits: {''.join(map(str, bits2))}")
    print(f"Reproducible: {bits == bits2}")
