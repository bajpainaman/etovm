use borsh::{BorshDeserialize, BorshSerialize};
use serde::{Deserialize, Serialize};

/// Rent sysvar - defines rent costs
#[derive(Clone, Debug, PartialEq, BorshSerialize, BorshDeserialize, Serialize, Deserialize)]
pub struct Rent {
    /// Rental rate in lamports/byte-year
    pub lamports_per_byte_year: u64,
    /// Amount of time (in years) a balance must be rent exempt
    pub exemption_threshold: f64,
    /// Account storage overhead for calculating base rent
    pub burn_percent: u8,
}

impl Default for Rent {
    fn default() -> Self {
        Self {
            lamports_per_byte_year: 3480,
            exemption_threshold: 2.0,
            burn_percent: 50,
        }
    }
}

impl Rent {
    /// Calculate minimum balance for rent exemption
    pub fn minimum_balance(&self, data_len: usize) -> u64 {
        let bytes = data_len + 128;
        ((bytes as f64) * self.lamports_per_byte_year as f64 * self.exemption_threshold) as u64
    }

    /// Check if an account is rent exempt
    pub fn is_exempt(&self, lamports: u64, data_len: usize) -> bool {
        lamports >= self.minimum_balance(data_len)
    }

    /// Calculate due rent for a given balance and data length
    pub fn due(&self, lamports: u64, data_len: usize, years_elapsed: f64) -> (u64, bool) {
        if self.is_exempt(lamports, data_len) {
            (0, true)
        } else {
            let rent_due =
                ((data_len + 128) as f64 * self.lamports_per_byte_year as f64 * years_elapsed)
                    as u64;
            (rent_due, false)
        }
    }

    pub fn to_account_data(&self) -> Vec<u8> {
        borsh::to_vec(self).expect("Failed to serialize Rent")
    }
}
