use borsh::{BorshDeserialize, BorshSerialize};
use serde::{Deserialize, Serialize};

/// Clock sysvar - provides time information
#[derive(
    Clone, Debug, Default, PartialEq, Eq, BorshSerialize, BorshDeserialize, Serialize, Deserialize,
)]
pub struct Clock {
    /// Current slot (maps to Avalanche block height)
    pub slot: u64,
    /// Epoch start timestamp
    pub epoch_start_timestamp: i64,
    /// Current epoch
    pub epoch: u64,
    /// Leader schedule epoch
    pub leader_schedule_epoch: u64,
    /// Unix timestamp (from Avalanche block)
    pub unix_timestamp: i64,
}

impl Clock {
    /// Create clock from Avalanche block context
    pub fn from_avalanche_block(height: u64, timestamp: i64, slots_per_epoch: u64) -> Self {
        let epoch = height / slots_per_epoch;
        let epoch_start_slot = epoch * slots_per_epoch;

        let slots_since_epoch_start = height - epoch_start_slot;
        let estimated_slot_duration_ms = 400;
        let epoch_start_timestamp =
            timestamp - (slots_since_epoch_start as i64 * estimated_slot_duration_ms / 1000);

        Self {
            slot: height,
            epoch_start_timestamp,
            epoch,
            leader_schedule_epoch: epoch + 1,
            unix_timestamp: timestamp,
        }
    }

    pub fn to_account_data(&self) -> Vec<u8> {
        borsh::to_vec(self).expect("Failed to serialize Clock")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clock_from_block() {
        let clock = Clock::from_avalanche_block(1000, 1700000000, 432000);
        assert_eq!(clock.slot, 1000);
        assert_eq!(clock.epoch, 0);
        assert_eq!(clock.unix_timestamp, 1700000000);
    }
}
