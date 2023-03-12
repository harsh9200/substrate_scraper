import pandas as pd
from typing import Dict
from substrateinterface import SubstrateInterface


def decode_scale(scale: str) -> str:
    return list(map(lambda x: x.value, scale))


class SubstrateScanner:
    def __init__(self, rpc_url: str, save_path: str = ".", save_df: bool = True):
        self.save_df = save_df
        self.save_path = save_path

        self.blocks_df_columns = [
            "BLOCK_PARENT_HASH",
            "BLOCK_NUMBER",
            "BLOCK_STATE_ROOT",
            "BLOCK_EXTRINSICS_ROOT",
            "BLOCK_DIGEST",
            "BLOCK_HASH",
            "BLOCK_EXTRINSICS",
        ]
        self.events_df_columns = [
            "EVENT_PHASE",
            "EVENT_EXTRINSIC_IDX",
            "EVENT_INDEX",
            "EVENT_MODULE_ID",
            "EVENT_EVENT_ID",
            "EVENT_ATTRIBUTES",
            "EVENT_TOPICS",
        ]
        self.extrinsics_df_columns = [
            "EXTRINSICS_HASH",
            "EXTRINSICS_LENGTH",
            "EXTRINSICS_CALL_INDEX",
            "EXTRINSICS_CALL_FUNCTION",
            "EXTRINSICS_CALL_MODULE",
            "EXTRINSICS_CALL_ARGS",
            "EXTRINSICS_CALL_HASH",
            "BLOCK_NUMBER",
        ]
        self.eras_stake_df_columns = [
            "ERA_IDX",
            "START_BLOCK_NUMBER",
            "END_BLOCK_NUMBER",
            "NUM_STAKERS",
            "TOTAL_STAKED",
            "TOTAL_REWARD",
            "INDIVIDUAL_STAKED",
        ]
        self.validators_df_columns = [
            "ERA_IDX",
            "VALIDATOR_ADDRESS",
            "NUMBER_OF_DELEGATORS",
            "DELEGATOR_ADDRESSES",
            "DELEGATION_AMOUNT",
            "DELEGATOR_STAKE",
            "INDIVIDUALLY_STAKE",
            "TOTAL_STAKED",
        ]

        self.substrate = SubstrateInterface(url=rpc_url)
        self.chain_name = self.substrate.chain

    def blocks(self, start_block: int = None, blocks_range: int = 5) -> pd.DataFrame:
        """Get the blocks related information on a given block range.

        Args:
            start_block: starting block number
            blocks_range: number of blocks to fetch the data from.

        Returns:
            DataFrame with substrate chain block information on provided blocks range
        """
        if not start_block:
            head_block = self.substrate.get_block_header()
            start_block = head_block["header"]["number"] - blocks_range

        blocks_df = pd.DataFrame(columns=self.blocks_df_columns)

        for idx, block_number in enumerate(
            range(start_block, start_block + blocks_range)
        ):
            block = self.substrate.get_block(block_number=block_number)
            blocks_df.loc[idx] = [
                block.get("header").get("parentHash"),
                block.get("header").get("number"),
                block.get("header").get("stateRoot"),
                block.get("header").get("extrinsicsRoot"),
                block.get("header").get("digest").get("logs"),
                block.get("header").get("hash"),
                block.get("extrinsics"),
            ]

        self.extrinsics(blocks_df=blocks_df)
        self.events(blocks_df=blocks_df)

        blocks_df["BLOCK_DIGEST"] = blocks_df["BLOCK_DIGEST"].apply(decode_scale)
        blocks_df.drop("BLOCK_EXTRINSICS", axis=1, inplace=True)

        if self.save_df:
            blocks_df.to_csv(
                f"{self.save_path}/{self.chain_name}-blocks.csv", index=False
            )
        return blocks_df

    def events(self, blocks_df: pd.DataFrame) -> pd.DataFrame:
        """Get the events related information on a given block range.

        Args:
            blocks_df: blocks dataframe

        Returns:
            DataFrame with substrate chain events information
        """
        events_df = pd.DataFrame(columns=self.events_df_columns)

        for _, block in blocks_df.iterrows():
            block_events = self.substrate.get_events(block.BLOCK_HASH)
            block_events = [event.value for event in block_events]

            event_df = pd.DataFrame(
                [
                    [
                        event.get("phase"),
                        event.get("extrinsic_idx"),
                        event.get("event_index"),
                        event.get("module_id"),
                        event.get("event_id"),
                        event.get("attributes"),
                        event.get("topics"),
                    ]
                    for event in block_events
                ],
                columns=self.events_df_columns,
            )
            events_df = pd.concat([events_df, event_df])

        if self.save_df:
            events_df.to_csv(
                f"{self.save_path}/{self.chain_name}-events.csv", index=False
            )
        return events_df

    def extrinsics(self, blocks_df: pd.DataFrame) -> pd.DataFrame:
        """Get the extrinsics related information.

        Args:
            blocks_df: blocks dataframe

        Returns:
            DataFrame with substrate chain extrinsics information
        """

        extrinsics_df = pd.DataFrame(columns=self.extrinsics_df_columns)

        for _, block in blocks_df.iterrows():
            block_extrinsics = block["BLOCK_EXTRINSICS"]
            block_extrinsics = [event.value for event in block_extrinsics]

            df = pd.DataFrame(
                [
                    [
                        extrinsic.get("extrinsic_hash"),
                        extrinsic.get("extrinsic_length"),
                        extrinsic.get("call").get("call_index"),
                        extrinsic.get("call").get("call_function"),
                        extrinsic.get("call").get("call_module"),
                        extrinsic.get("call").get("call_args"),
                        extrinsic.get("call").get("call_hash"),
                        block.BLOCK_NUMBER,
                    ]
                    for extrinsic in block_extrinsics
                ],
                columns=self.extrinsics_df_columns,
            )
            extrinsics_df = pd.concat([extrinsics_df, df])

        if self.save_df:
            extrinsics_df.to_csv(
                f"{self.save_path}/{self.chain_name}-extrinsics.csv", index=False
            )
        return extrinsics_df

    def eras_staking(self, start_era: int = None, eras_range: int = 10) -> pd.DataFrame:
        """Get the eras related information.

        Args:
            start_era: starting era index
            eras_range: number of eras to fetch the data from.

        Returns:
            DataFrame with substrate chain eras information
        """

        if not start_era:
            start_era = (
                self.substrate.query(
                    module="Staking",
                    storage_function="CurrentEra",
                ).value
                - eras_range
            )

        eras_stake_df = pd.DataFrame(columns=self.eras_stake_df_columns)
        eras_stat_dict = self.get_era_stats(start_era)

        for idx, era_idx in enumerate(range(start_era, start_era + eras_range + 1)):
            block_hash = self.get_block_hash(eras_stat_dict.get(start_era, [None])[0])

            eras_total_staked = self.substrate.query(
                module="Staking",
                storage_function="ErasTotalStake",
                params=[era_idx],
                block_hash=block_hash,
            )

            eras_reward_points = self.substrate.query(
                module="Staking",
                storage_function="ErasRewardPoints",
                params=[era_idx],
                block_hash=block_hash,
            )

            eras_stake_df.loc[idx] = [
                era_idx,
                eras_stat_dict.get(era_idx)[0],
                eras_stat_dict.get(era_idx)[1],
                len(eras_reward_points.value.get("individual")),
                eras_total_staked,
                int(eras_reward_points.value.get("total")),
                eras_reward_points.value.get("individual"),
            ]

        if self.save_df:
            eras_stake_df.to_csv(
                f"{self.save_path}/{self.chain_name}-eras_staked.csv", index=False
            )

        return eras_stake_df

    def validators(self, eras_stake_df: pd.DataFrame) -> pd.DataFrame:
        """Get the validators related information.

        Args:
            eras_stake_df: dataframe with eras information

        Returns:
            DataFrame with substrate chain validators information
        """

        validators_df = pd.DataFrame(columns=self.validators_df_columns)

        for idx, individual_validator in eras_stake_df.iterrows():
            era_idx = individual_validator.get("ERA_IDX")
            validators_list = individual_validator.get("INDIVIDUAL_STAKED")

            for _, validator_info in enumerate(validators_list[:10]):
                validator_address, _ = validator_info

                response = self.substrate.query(
                    module="Staking",
                    storage_function="ErasStakers",
                    params=[
                        era_idx,
                        validator_address,
                    ],
                )
                response = response.value
                total_staked, individually_staked = response["total"], response["own"]

                validators_df.loc[idx] = [
                    era_idx,
                    validator_address,
                    len(response.get("others", [])),
                    list(map(lambda x: x.get("who"), response.get("others"))),
                    list(map(lambda x: x.get("value") / 1e10, response.get("others"))),
                    total_staked - individually_staked,
                    individually_staked,
                    total_staked,
                ]

        if self.save_df:
            validators_df.to_csv(
                f"{self.save_path}/{self.chain_name}-validators.csv", index=False
            )

        return validators_df

    def get_block_hash(self, block_number=None):
        head_block = self.substrate.get_block(block_number=block_number)
        block_hash = self.substrate.get_block_hash(
            block_id=head_block["header"]["number"]
        )

        return block_hash

    def get_era_stats(self, target_era_idx) -> Dict[str, list]:
        current_era_idx = self.substrate.query(
            module="Staking",
            storage_function="CurrentEra",
        ).value

        start_session_epoch = self.substrate.query(
            module="Staking",
            storage_function="ErasStartSessionIndex",
            params=[current_era_idx],
        ).value

        current_epoch = self.substrate.query(
            module="Session",
            storage_function="CurrentIndex",
        ).value

        target_epoch = (
            start_session_epoch - ((current_era_idx - target_era_idx) * 3) - 1
        )

        era_stats = {}
        previous_start_block = None
        block_hash = self.get_block_hash()

        for epoch in range(current_epoch - 1, target_epoch, -1):
            start_block, _ = self.substrate.query(
                module="Babe", storage_function="EpochStart", block_hash=block_hash
            ).value

            block_hash = self.substrate.get_block_hash(block_id=start_block - 1)

            if (epoch / 3).is_integer():
                era = self.substrate.query(
                    module="Staking",
                    storage_function="CurrentEra",
                    block_hash=self.substrate.get_block_hash(block_id=start_block),
                ).value

                era_stats[era] = [start_block, previous_start_block]
                previous_start_block = start_block - 1

        return era_stats


if __name__ == "__main__":
    rpc_url = "wss://rpc.polkadot.io"
    # rpc_url = "wss://kusama-rpc.polkadot.io"

    polkadot_interface = SubstrateScanner(rpc_url)
    polkadot_interface.era_staked_total(start_era=1010, eras_range=5)
