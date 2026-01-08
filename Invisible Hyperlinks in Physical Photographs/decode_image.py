# import bchlib
# import glob
# from PIL import Image, ImageOps
# import numpy as np
# import tensorflow as tf
# # 🔸 Removed: tensorflow.contrib.image (not used and unavailable in TF2)
# from tensorflow.python.saved_model import tag_constants
# from tensorflow.python.saved_model import signature_constants

# BCH_POLYNOMIAL = 137
# BCH_BITS = 5

# def encode_original_secret(secret_str):
#     """Encode original secret string into 96-bit list (same as encoder)."""
#     if len(secret_str) > 7:
#         raise ValueError("Secret must be ≤7 characters")
#     padded = secret_str.ljust(7)
#     bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
#     data = bytearray(padded, 'utf-8')
#     ecc = bch.encode(data)
#     packet = data + ecc
#     packet_binary = ''.join(format(x, '08b') for x in packet)
#     return [int(bit) for bit in packet_binary]  # 96 bits

# def main():
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('model', type=str)
#     parser.add_argument('--image', type=str, default=None)
#     parser.add_argument('--images_dir', type=str, default=None)
#     parser.add_argument('--secret_size', type=int, default=100)
#     parser.add_argument('--original_secret', type=str, default=None,
#                     help='Original secret to compute bit accuracy')
#     args = parser.parse_args()

#     if args.image is not None:
#         files_list = [args.image]
#     elif args.images_dir is not None:
#         files_list = glob.glob(args.images_dir + '/*')
#     else:
#         print('Missing input image')
#         return

#     # ✅ tf.compat.v1: InteractiveSession
#     sess = tf.compat.v1.InteractiveSession(graph=tf.Graph())

#     # ✅ tf.compat.v1: saved_model.loader.load
#     model = tf.compat.v1.saved_model.loader.load(sess, [tag_constants.SERVING], args.model)

#     input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    
#     # ✅ tf.compat.v1: get_default_graph
#     input_image = tf.compat.v1.get_default_graph().get_tensor_by_name(input_image_name)

#     output_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['decoded'].name
#     output_secret = tf.compat.v1.get_default_graph().get_tensor_by_name(output_secret_name)

#     bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

#     for filename in files_list:
#         image = Image.open(filename).convert("RGB")
#         image = np.array(ImageOps.fit(image, (400, 400)), dtype=np.float32)
#         image /= 255.

#         feed_dict = {input_image: [image]}

#         secret = sess.run([output_secret], feed_dict=feed_dict)[0][0]

#         recovered_bits = [...]  # first 96 bits from model output

#         # Get original bits
#         if args.original_secret:
#             original_bits = encode_secret(args.original_secret)
    
#         # Compute bit accuracy
#         correct_bits = sum(a == b for a, b in zip(original_bits, recovered_bits))
#         bit_accuracy = correct_bits / len(original_bits)
#         print(f"Bit Accuracy: {bit_accuracy:.4f} ({correct_bits}/{len(original_bits)})")

#         packet_binary = "".join([str(int(bit)) for bit in secret[:96]])
#         packet = bytes(int(packet_binary[i : i + 8], 2) for i in range(0, len(packet_binary), 8))
#         packet = bytearray(packet)

#         data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]

#         bitflips = bch.decode_inplace(data, ecc)

#         if bitflips != -1:
#             try:
#                 code = data.decode("utf-8")
#                 print(filename, code)
#                 continue
#             except:
#                 continue
#         print(filename, 'Failed to decode')


# if __name__ == "__main__":
#     # ✅ Critical: disable eager execution for TF 2.x
#     tf.compat.v1.disable_eager_execution()
#     main()

import bchlib
import glob
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

BCH_POLYNOMIAL = 137
BCH_BITS = 5

def encode_original_secret(secret_str):
    """Encode original secret string into 96-bit list (same as encoder)."""
    if len(secret_str) > 7:
        raise ValueError("Secret must be ≤7 characters")
    padded = secret_str.ljust(7)
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    data = bytearray(padded, 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc
    packet_binary = ''.join(format(x, '08b') for x in packet)
    return [int(bit) for bit in packet_binary]  # 96 bits

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default=None)
    parser.add_argument('--secret_size', type=int, default=100)
    parser.add_argument('--original_secret', type=str, default=None,
                        help='Original secret to compute bit accuracy')
    args = parser.parse_args()

    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(args.images_dir + '/*')
    else:
        print('Missing input image')
        return

    sess = tf.compat.v1.InteractiveSession(graph=tf.Graph())
    model = tf.compat.v1.saved_model.loader.load(sess, [tag_constants.SERVING], args.model)

    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_image = tf.compat.v1.get_default_graph().get_tensor_by_name(input_image_name)

    output_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['decoded'].name
    output_secret = tf.compat.v1.get_default_graph().get_tensor_by_name(output_secret_name)

    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    for filename in files_list:
        image = Image.open(filename).convert("RGB")
        image = np.array(ImageOps.fit(image, (400, 400)), dtype=np.float32)
        image /= 255.

        feed_dict = {input_image: [image]}
        secret = sess.run([output_secret], feed_dict=feed_dict)[0][0]

        recovered_bits = [int(bit) for bit in secret[:96]]

        if args.original_secret is not None:
            try:
                original_bits = encode_original_secret(args.original_secret)
                correct = sum(a == b for a, b in zip(original_bits, recovered_bits))
                bit_acc = correct / len(original_bits)
                print(f"Bit Accuracy: {bit_acc:.4f} ({correct}/{len(original_bits)})")
            except Exception as e:
                print(f"⚠️ Bit accuracy error: {e}")

        packet_binary = "".join([str(int(bit)) for bit in secret[:96]])
        packet = bytes(int(packet_binary[i : i + 8], 2) for i in range(0, len(packet_binary), 8))
        packet = bytearray(packet)

        data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]
        bitflips = bch.decode_inplace(data, ecc)

        if bitflips != -1:
            try:
                code = data.decode("utf-8")
                print(filename, code)
                continue
            except:
                continue
        print(filename, 'Failed to decode')

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    main()
