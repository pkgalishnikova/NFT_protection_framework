import { getContract } from "thirdweb";
import { sepolia } from "thirdweb/chains";
import { client } from "./app/client";

export const CONTRACT_ADDRESS = "0x31c5B73bf8c813570A34DB2f531cd663fA1B7EaD";

export const verifyNFTContract = getContract({
  client,
  chain: sepolia,
  address: CONTRACT_ADDRESS,
});