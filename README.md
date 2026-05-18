# NFT protection framework

NFTs offer many benefits, however they also raise concerns regarding the authenticity of digital assets. NFT images are often uploaded to websites or marketplaces where others can easily download, copy, and re-mint them as their own. Unfortunately, current blockchain systems and external storage platforms make such actions possible. This work describes the development of a framework for NFT authenticity protection and validation. It provides a convenient way for users to perform NFT watermarking and extraction. A web application was created for the users who have no coding experience. It allows to upload image, embed wallet address in it and later verify the ownership through user interface. The backend functionality was implemented through the smart contract and HiDDeN-based encoder-decoder, which handle on-chain NFT minting and watermark embedding/extraction respectively. The work showed promising results in watermark embedding and decoding, proving that the model is robust and can recover hidden messages with high bit accuracy.

## Structure

- `model/` — Watermark embedding and decoding models (HiDDeN and StegaStamp options)
- `smart_contract/` — Solidity contract for NFT minting
- `web_application/` — User interface for the NFT verification framework

## Pipeline

1. The prefix of owner's Ethereum wallet address is embedded invisibly into the NFT image by the HiDDeN encoder.
2. The smart contract mints the NFT on-chain with an NFT image using confirmation on MetaMask.
3. To verify, the HiDDeN decoder extracts the string from the image (trying multiple variants for robustness) and compares the recovered wallet prefix against the claimed owner.
