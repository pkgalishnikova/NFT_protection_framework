def ethereum_to_bits(address, num_bits=100):
    if address.startswith('0x') or address.startswith('0X'):
        address = address[2:]

    num_hex_chars = num_bits // 4
    address_part = address[:num_hex_chars].upper()

    binary_str = bin(int(address_part, 16))[2:].zfill(num_bits)
    bits = torch.tensor([int(b) for b in binary_str], dtype=torch.float32)

    return bits


def bits_to_ethereum(bits, num_bits=100, original_secret=SECRET_SHORT):
    bits_np = (bits[:num_bits] > 0.5).cpu().numpy().astype(np.uint8)
    binary_str = ''.join([str(int(b)) for b in bits_np])

    try:
        hex_value = hex(int(binary_str, 2))[2:].upper()
        num_hex_chars = num_bits // 4  # 100 bits = 25 hex chars
        hex_value = hex_value.zfill(num_hex_chars)

        original_hex_len = len(original_secret) - 2
        return '0x' + hex_value[:original_hex_len]
    except:
        original_hex_len = len(original_secret) - 2
        return "0x" + "?"*original_hex_len

SECRET_STR = "0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D"
print(f"Full address: {SECRET_STR}")

SECRET_SHORT = SECRET_STR[:12]  # "0xBC4CA0EdA7"
MESSAGE_LEN = 100  # bits

print(f"📊 Using SHORT version: '{SECRET_SHORT}' ({MESSAGE_LEN} bits)")
