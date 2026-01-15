use borsh::{BorshDeserialize, BorshSerialize};
use serde::{Deserialize, Serialize};

/// Epoch schedule sysvar
#[derive(
    Clone, Debug, PartialEq, Eq, BorshSerialize, BorshDeserialize, Serialize, Deserialize,
)]
pub struct EpochSchedule {
    /// Slots per epoch
    pub slots_per_epoch: u64,
    /// Number of slots before epoch starts to calculate leader schedule
    pub leader_schedule_slot_offset: u64,
    /// Whether epochs start short and grow
    pub warmup: bool,
    /// First normal-length epoch
    pub first_normal_epoch: u64,
    /// First slot of first_normal_epoch
    pub first_normal_slot: u64,
}

impl Default for EpochSchedule {
    fn default() -> Self {
        Self {
            slots_per_epoch: 432000,
            leader_schedule_slot_offset: 432000,
            warmup: false,
            first_normal_epoch: 0,
            first_normal_slot: 0,
        }
    }
}

impl EpochSchedule {
    /// Get the epoch for a given slot
    pub fn get_epoch(&self, slot: u64) -> u64 {
        slot / self.slots_per_epoch
    }

    /// Get the first slot of a given epoch
    pub fn get_first_slot_in_epoch(&self, epoch: u64) -> u64 {
        epoch * self.slots_per_epoch
    }

    /// Get the last slot of a given epoch
    pub fn get_last_slot_in_epoch(&self, epoch: u64) -> u64 {
        self.get_first_slot_in_epoch(epoch + 1) - 1
    }

    pub fn to_account_data(&self) -> Vec<u8> {
        borsh::to_vec(self).expect("Failed to serialize EpochSchedule")
    }
}
