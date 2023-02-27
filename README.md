# eth-rl

嘿嘿嘿
啵唧啵唧啵唧啵唧啵唧啵唧

## Block Lifecycle

References
1. <https://medium.com/coinmonks/ethereum-2-0-c7175646a0b3#:~:text=How%20PoS%20works%20on%20Ethereum%201%20Blocks%20can,attestations%20%28voting%20for%20the%20main%20chain%29.%20More%20items>

- The network randomly selects one validator to be the proposer for each slot.
- If the assigned validator misses the opportunity to propose a block, the network will have no block for that slot and progresses to the next slot.
    - If the validator is honest, the block got proposed; if the validator is malicious, it will not propose the block. (In this implementation we directly select proposer in honest validators.)
- During each slot, validators will take turns submitting attestations (voting for the main chain).
    - The block got proposed when 50% of the validators vote to the block.
- The votes determine the blocks for every epoch (one epoch is 6.4 minutes, consisting of 32 slots).
- Each validator submits their attestations once every epoch.
- Finality requires at least 2 epochs (≈ 12.8 minutes).
- Validators will also be monitoring each other for malicious behaviors.
- If they observe another validator propose two blocks in the same slot or submit attestation votes that contradict themselves, they can alert the - network.
- The network will reward the whistleblower and will slash the violator.
- Rewards, penalties, and slashing are processed every epoch. Inflationary rewards are when validators are issued ETH for conducting the work. - Properly submitting attestations and including other validator attestations when proposing blocks yields a validator anywhere from 2% to 22% in - staking rewards, dependent on how much of the entire network is staked.
- If a validator fails to stay online and execute their share of computational responsibilities, their block reward will moderately decrease (lose - 67K Gwei for each epoch they are offline for) in order to incentivize validators to stay online as consistently as possible. The penalty amounts - have been intentionally set low so that honest validators with low connectivity can still be netting positive staking rewards.
