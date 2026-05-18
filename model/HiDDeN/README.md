# HiDDeN Model

The core watermarking model used in the framework, based on the HiDDeN architecture (Zhu et al., 2018) and tuned for the NFT watermarking task.

The encoder embeds a 32-bit message invisibly into a 256×256 image. The decoder recovers the bit string from an image, which can be possibly damaged by image attack.

## Files

- `model.py`: Code of the Encoder-Decoder model, used in web interface.
- `checkpoints/`: Checkpoints saved as a result of the model's work. 
- `HiDDeN_experiments/`: Modifications, metrics and outputs from training and evaluation variations of a model. See the README inside for a description of each experiment.
