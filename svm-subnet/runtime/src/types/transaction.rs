use super::{AccountMeta, CompiledInstruction, Instruction, Pubkey};
use borsh::{BorshDeserialize, BorshSerialize};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use sha2::{Digest, Sha256};

/// A Solana-compatible transaction
#[derive(Clone, Debug, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct Transaction {
    /// Signatures for this transaction
    pub signatures: Vec<[u8; 64]>,
    /// The message containing instructions
    pub message: Message,
}

/// The message component of a transaction
#[derive(Clone, Debug, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct Message {
    /// Message header with signer counts
    pub header: MessageHeader,
    /// All account pubkeys used in this transaction
    pub account_keys: Vec<Pubkey>,
    /// Recent blockhash for replay protection
    pub recent_blockhash: [u8; 32],
    /// Instructions to execute
    pub instructions: Vec<CompiledInstruction>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct MessageHeader {
    /// Number of required signatures
    pub num_required_signatures: u8,
    /// Number of read-only signed accounts
    pub num_readonly_signed_accounts: u8,
    /// Number of read-only unsigned accounts
    pub num_readonly_unsigned_accounts: u8,
}

impl Transaction {
    pub fn new(message: Message, signers: &[&ed25519_dalek::SigningKey]) -> Self {
        let message_bytes = borsh::to_vec(&message).expect("Failed to serialize message");

        let signatures: Vec<[u8; 64]> = signers
            .iter()
            .map(|signer| {
                use ed25519_dalek::Signer;
                signer.sign(&message_bytes).to_bytes()
            })
            .collect();

        Self { signatures, message }
    }

    pub fn new_unsigned(message: Message) -> Self {
        let num_signatures = message.header.num_required_signatures as usize;
        Self {
            signatures: vec![[0u8; 64]; num_signatures],
            message,
        }
    }

    /// Verify all signatures
    pub fn verify(&self) -> Result<(), TransactionError> {
        let message_bytes =
            borsh::to_vec(&self.message).map_err(|_| TransactionError::SerializationError)?;

        let num_signers = self.message.header.num_required_signatures as usize;

        if self.signatures.len() != num_signers {
            return Err(TransactionError::InvalidSignatureCount);
        }

        for (i, sig_bytes) in self.signatures.iter().enumerate() {
            let pubkey = &self.message.account_keys[i];
            let signature = Signature::from_bytes(sig_bytes);
            let verifying_key = VerifyingKey::from_bytes(pubkey.as_ref())
                .map_err(|_| TransactionError::InvalidPublicKey)?;

            verifying_key
                .verify(&message_bytes, &signature)
                .map_err(|_| TransactionError::InvalidSignature)?;
        }

        Ok(())
    }

    /// Get the transaction signature (first signature)
    pub fn signature(&self) -> Option<&[u8; 64]> {
        self.signatures.first()
    }

    /// Compute the transaction hash
    pub fn hash(&self) -> [u8; 32] {
        let bytes = borsh::to_vec(self).expect("Failed to serialize transaction");
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Get all signers
    pub fn signers(&self) -> &[Pubkey] {
        let num_signers = self.message.header.num_required_signatures as usize;
        &self.message.account_keys[..num_signers]
    }

    /// Get the fee payer (first signer)
    pub fn fee_payer(&self) -> Option<&Pubkey> {
        self.message.account_keys.first()
    }
}

impl Message {
    pub fn new(instructions: &[Instruction], payer: Option<&Pubkey>) -> Self {
        let mut signed_keys = Vec::new();
        let mut unsigned_keys = Vec::new();
        let mut readonly_signed = Vec::new();
        let mut readonly_unsigned = Vec::new();

        // Add payer first (always signer and writable)
        if let Some(payer) = payer {
            signed_keys.push(*payer);
        }

        // Collect all accounts from instructions
        for ix in instructions {
            for account in &ix.accounts {
                if account.is_signer {
                    if account.is_writable {
                        if !signed_keys.contains(&account.pubkey) {
                            signed_keys.push(account.pubkey);
                        }
                    } else if !readonly_signed.contains(&account.pubkey) {
                        readonly_signed.push(account.pubkey);
                    }
                } else if account.is_writable {
                    if !unsigned_keys.contains(&account.pubkey) {
                        unsigned_keys.push(account.pubkey);
                    }
                } else if !readonly_unsigned.contains(&account.pubkey) {
                    readonly_unsigned.push(account.pubkey);
                }
            }

            // Add program ID as readonly unsigned
            if !readonly_unsigned.contains(&ix.program_id) {
                readonly_unsigned.push(ix.program_id);
            }
        }

        // Build account keys in proper order
        let mut account_keys = Vec::new();
        account_keys.extend(signed_keys.iter());
        account_keys.extend(readonly_signed.iter());
        account_keys.extend(unsigned_keys.iter());
        account_keys.extend(readonly_unsigned.iter());

        // Compile instructions
        let compiled_instructions: Vec<CompiledInstruction> = instructions
            .iter()
            .map(|ix| {
                let program_id_index = account_keys
                    .iter()
                    .position(|k| k == &ix.program_id)
                    .expect("Program ID not in account keys") as u8;

                let accounts: Vec<u8> = ix
                    .accounts
                    .iter()
                    .map(|meta| {
                        account_keys
                            .iter()
                            .position(|k| k == &meta.pubkey)
                            .expect("Account not in account keys") as u8
                    })
                    .collect();

                CompiledInstruction {
                    program_id_index,
                    accounts,
                    data: ix.data.clone(),
                }
            })
            .collect();

        let header = MessageHeader {
            num_required_signatures: signed_keys.len() as u8,
            num_readonly_signed_accounts: readonly_signed.len() as u8,
            num_readonly_unsigned_accounts: readonly_unsigned.len() as u8,
        };

        Self {
            header,
            account_keys,
            recent_blockhash: [0u8; 32],
            instructions: compiled_instructions,
        }
    }

    pub fn set_recent_blockhash(&mut self, blockhash: [u8; 32]) {
        self.recent_blockhash = blockhash;
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TransactionError {
    #[error("Invalid signature")]
    InvalidSignature,
    #[error("Invalid signature count")]
    InvalidSignatureCount,
    #[error("Invalid public key")]
    InvalidPublicKey,
    #[error("Serialization error")]
    SerializationError,
    #[error("Account not found: {0}")]
    AccountNotFound(Pubkey),
    #[error("Insufficient funds for fee")]
    InsufficientFundsForFee,
    #[error("Invalid account for fee")]
    InvalidAccountForFee,
    #[error("Already processed")]
    AlreadyProcessed,
    #[error("Blockhash not found")]
    BlockhashNotFound,
    #[error("Instruction error: {0} - {1}")]
    InstructionError(u8, InstructionError),
}

#[derive(Debug, thiserror::Error)]
pub enum InstructionError {
    #[error("Generic error")]
    GenericError,
    #[error("Invalid argument")]
    InvalidArgument,
    #[error("Invalid instruction data")]
    InvalidInstructionData,
    #[error("Invalid account data")]
    InvalidAccountData,
    #[error("Account data too small")]
    AccountDataTooSmall,
    #[error("Insufficient funds")]
    InsufficientFunds,
    #[error("Incorrect program id")]
    IncorrectProgramId,
    #[error("Missing required signature")]
    MissingRequiredSignature,
    #[error("Account already initialized")]
    AccountAlreadyInitialized,
    #[error("Uninitialized account")]
    UninitializedAccount,
    #[error("Custom error: {0}")]
    Custom(u32),
}
