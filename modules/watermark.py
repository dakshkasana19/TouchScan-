import numpy as np


def string_to_bits(text):
    encoded_bytes = text.encode('utf-8')
    length_header = len(encoded_bytes).to_bytes(4, byteorder='big')
    full_payload = length_header + encoded_bytes
    bits = []
    for byte in full_payload:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def bits_to_string(bits):
    if len(bits) < 32:
        raise ValueError("Bit stream too short to read length header.")

    length_bytes = []
    for byte_idx in range(4):
        byte_val = 0
        for bit_idx in range(8):
            byte_val = (byte_val << 1) | bits[byte_idx * 8 + bit_idx]
        length_bytes.append(byte_val)

    payload_length = int.from_bytes(bytes(length_bytes), byteorder='big')
    total_bits_needed = 32 + (payload_length * 8)
    if len(bits) < total_bits_needed:
        raise ValueError(f"Incomplete payload: expected {total_bits_needed} bits, got {len(bits)} bits.")

    payload_bytes = []
    for byte_idx in range(payload_length):
        start = 32 + byte_idx * 8
        byte_val = 0
        for bit_idx in range(8):
            byte_val = (byte_val << 1) | bits[start + bit_idx]
        payload_bytes.append(byte_val)

    try:
        return bytes(payload_bytes).decode('utf-8')
    except UnicodeDecodeError as exc:
        raise ValueError("Decoded watermark payload is not valid UTF-8 text.") from exc


def get_available_capacity(image):
    if len(image.shape) != 2:
        raise ValueError("Image must be single-channel grayscale.")
    eligible_pixels = np.count_nonzero((image == 255) | (image == 254))
    available_chars = max(0, (eligible_pixels - 32) // 8)
    return eligible_pixels, available_chars


def embed_watermark(image, watermark_text):
    if len(image.shape) != 2:
        raise ValueError("Image must be single-channel grayscale.")
    if not isinstance(watermark_text, str) or len(watermark_text) == 0:
        raise ValueError("Watermark text must be a non-empty string.")

    bits_to_embed = string_to_bits(watermark_text)
    total_bits = len(bits_to_embed)

    white_pixel_coords = np.argwhere(image == 255)
    total_white_pixels = len(white_pixel_coords)

    if total_white_pixels < total_bits:
        available_chars = max(0, (total_white_pixels - 32) // 8)
        raise ValueError(
            f"Payload exceeds available capacity: requires {total_bits} bits ({len(watermark_text.encode('utf-8'))} bytes), "
            f"but only {total_white_pixels} white background pixels are available (max capacity ~{available_chars} chars)."
        )

    watermarked_image = image.copy()
    for idx, bit in enumerate(bits_to_embed):
        r, c = white_pixel_coords[idx]
        if bit == 1:
            watermarked_image[r, c] = 254
        else:
            watermarked_image[r, c] = 255

    return watermarked_image, total_bits


def extract_watermark(image):
    if len(image.shape) != 2:
        raise ValueError("Image must be single-channel grayscale.")

    candidate_coords = np.argwhere((image == 255) | (image == 254))
    if len(candidate_coords) < 32:
        raise ValueError("Insufficient background pixels to detect watermark header.")

    extracted_bits = []
    for r, c in candidate_coords:
        pixel_val = image[r, c]
        if pixel_val == 254:
            extracted_bits.append(1)
        elif pixel_val == 255:
            extracted_bits.append(0)

        if len(extracted_bits) == 32:
            length_bytes = []
            for b_idx in range(4):
                val = 0
                for bit_pos in range(8):
                    val = (val << 1) | extracted_bits[b_idx * 8 + bit_pos]
                length_bytes.append(val)
            expected_payload_length = int.from_bytes(bytes(length_bytes), byteorder='big')
            if expected_payload_length < 0 or expected_payload_length > 100000:
                raise ValueError(f"No valid watermark detected (implausible payload length: {expected_payload_length}).")
            target_total_bits = 32 + (expected_payload_length * 8)
            if len(candidate_coords) < target_total_bits:
                raise ValueError("Image lacks enough candidate pixels for stated watermark payload.")

        if 'target_total_bits' in locals() and len(extracted_bits) == target_total_bits:
            break

    return bits_to_string(extracted_bits)

