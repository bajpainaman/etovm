use borsh::{BorshDeserialize, BorshSerialize};
use serde::{Deserialize, Serialize};

/// Fee calculator for transactions
#[derive(
    Clone, Debug, Default, PartialEq, Eq, BorshSerialize, BorshDeserialize, Serialize, Deserialize,
)]
pub struct FeeCalculator {
    /// Lamports per signature
    pub lamports_per_signature: u64,
}

impl FeeCalculator {
    pub fn new(lamports_per_signature: u64) -> Self {
        Self {
            lamports_per_signature,
        }
    }

    /// Calculate fee for a transaction
    pub fn calculate_fee(&self, num_signatures: usize) -> u64 {
        self.lamports_per_signature * num_signatures as u64
    }
}

/// Fees sysvar
#[derive(
    Clone, Debug, Default, PartialEq, Eq, BorshSerialize, BorshDeserialize, Serialize, Deserialize,
)]
pub struct Fees {
    pub fee_calculator: FeeCalculator,
}

impl Fees {
    pub fn new(fee_calculator: FeeCalculator) -> Self {
        Self { fee_calculator }
    }

    pub fn to_account_data(&self) -> Vec<u8> {
        borsh::to_vec(self).expect("Failed to serialize Fees")
    }
}
