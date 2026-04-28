// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Strings.sol";

/**
 * @title VerifyNFT
 * @notice ERC-721 contract for watermarked NFTs.
 */
contract VerifyNFT is ERC721URIStorage, Ownable {
    using Strings for uint256;

    uint256 private _tokenIds;

    mapping(uint256 => string) public embeddedAddress;

    event NFTMinted(
        uint256 indexed tokenId,
        address indexed owner,
        string tokenURI,
        string embeddedAddr
    );

    constructor() ERC721("VerifyNFT", "VNFT") Ownable(msg.sender) {}

    function mint(
        string memory tokenURI_,
        string memory embeddedAddr
    ) public returns (uint256) {
        uint256 newTokenId = _tokenIds;

        _mint(msg.sender, newTokenId);
        _setTokenURI(newTokenId, tokenURI_);
        embeddedAddress[newTokenId] = embeddedAddr;

        emit NFTMinted(newTokenId, msg.sender, tokenURI_, embeddedAddr);

        _tokenIds += 1;
        return newTokenId;
    }

    function tokensOfOwner(
        address owner
    ) public view returns (uint256[] memory) {
        uint256 balance = balanceOf(owner);
        uint256[] memory tokenIds = new uint256[](balance);
        uint256 count = 0;
        for (uint256 i = 0; i < _tokenIds; i++) {
            try this.ownerOf(i) returns (address tokenOwner) {
                if (tokenOwner == owner) {
                    tokenIds[count] = i;
                    count++;
                }
            } catch {}
        }
        return tokenIds;
    }

    function getEmbeddedAddress(
        uint256 tokenId
    ) public view returns (string memory) {
        return embeddedAddress[tokenId];
    }

    function totalMinted() public view returns (uint256) {
        return _tokenIds;
    }
}
