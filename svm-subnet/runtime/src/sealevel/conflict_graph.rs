//! Conflict Graph - Transaction Dependency Analysis
//!
//! Builds a directed graph where edges represent dependencies between transactions.
//! Used to determine which transactions can execute in parallel.

use super::AccessSet;
use crate::types::Pubkey;
use std::collections::{HashMap, HashSet};

/// Index of a transaction in the batch
pub type TxIndex = usize;

/// Conflict graph for a set of transactions
///
/// Represents dependencies between transactions. An edge from tx_a to tx_b
/// means tx_a must complete before tx_b can start.
#[derive(Debug)]
pub struct ConflictGraph {
    /// Number of transactions
    pub num_transactions: usize,
    /// Adjacency list: tx_index -> set of dependent tx indices
    pub dependencies: Vec<HashSet<TxIndex>>,
    /// Reverse adjacency: tx_index -> set of tx indices that depend on it
    pub dependents: Vec<HashSet<TxIndex>>,
    /// In-degree count for each transaction (for topological sort)
    pub in_degree: Vec<usize>,
    /// Map from account to last writer transaction index
    last_writer: HashMap<Pubkey, TxIndex>,
    /// Map from account to set of reader transaction indices since last write
    readers_since_write: HashMap<Pubkey, HashSet<TxIndex>>,
}

impl ConflictGraph {
    /// Create a new conflict graph for N transactions
    pub fn new(num_transactions: usize) -> Self {
        Self {
            num_transactions,
            dependencies: vec![HashSet::new(); num_transactions],
            dependents: vec![HashSet::new(); num_transactions],
            in_degree: vec![0; num_transactions],
            last_writer: HashMap::new(),
            readers_since_write: HashMap::new(),
        }
    }

    /// Build conflict graph from a list of access sets
    ///
    /// Returns the graph with all dependencies computed.
    pub fn from_access_sets(access_sets: &[AccessSet]) -> Self {
        let mut graph = Self::new(access_sets.len());

        for (tx_idx, access) in access_sets.iter().enumerate() {
            graph.add_transaction(tx_idx, access);
        }

        graph
    }

    /// Add a transaction's access set to the graph
    fn add_transaction(&mut self, tx_idx: TxIndex, access: &AccessSet) {
        // Handle reads - must wait for any previous writer
        for pubkey in &access.reads {
            if let Some(&writer_idx) = self.last_writer.get(pubkey) {
                if writer_idx != tx_idx {
                    self.add_dependency(tx_idx, writer_idx);
                }
            }
        }

        // Handle writes - must wait for previous writer AND all readers
        for pubkey in &access.writes {
            // Depend on last writer
            if let Some(&writer_idx) = self.last_writer.get(pubkey) {
                if writer_idx != tx_idx {
                    self.add_dependency(tx_idx, writer_idx);
                }
            }

            // Collect readers first to avoid borrow conflicts
            let readers_to_depend: Vec<TxIndex> = self
                .readers_since_write
                .get(pubkey)
                .map(|readers| {
                    readers
                        .iter()
                        .copied()
                        .filter(|&r| r != tx_idx)
                        .collect()
                })
                .unwrap_or_default();

            // Depend on all readers since last write
            for reader_idx in readers_to_depend {
                self.add_dependency(tx_idx, reader_idx);
            }

            // Update last writer and clear readers
            self.last_writer.insert(*pubkey, tx_idx);
            self.readers_since_write.remove(pubkey);
        }

        // Track reads for future write dependencies
        for pubkey in &access.reads {
            self.readers_since_write
                .entry(*pubkey)
                .or_default()
                .insert(tx_idx);
        }
    }

    /// Add a dependency: tx_idx depends on dep_idx
    fn add_dependency(&mut self, tx_idx: TxIndex, dep_idx: TxIndex) {
        if self.dependencies[tx_idx].insert(dep_idx) {
            self.dependents[dep_idx].insert(tx_idx);
            self.in_degree[tx_idx] += 1;
        }
    }

    /// Get transactions with no dependencies (can execute immediately)
    pub fn get_ready_transactions(&self) -> Vec<TxIndex> {
        self.in_degree
            .iter()
            .enumerate()
            .filter_map(|(idx, &deg)| if deg == 0 { Some(idx) } else { None })
            .collect()
    }

    /// Check if a specific transaction is ready (no dependencies)
    pub fn is_ready(&self, tx_idx: TxIndex) -> bool {
        self.in_degree.get(tx_idx).map_or(false, |&d| d == 0)
    }

    /// Get all transactions that depend on the given transaction
    pub fn get_dependents(&self, tx_idx: TxIndex) -> &HashSet<TxIndex> {
        &self.dependents[tx_idx]
    }

    /// Get all dependencies for a transaction
    pub fn get_dependencies(&self, tx_idx: TxIndex) -> &HashSet<TxIndex> {
        &self.dependencies[tx_idx]
    }

    /// Check if the graph has any cycles (should never happen with proper ordering)
    pub fn has_cycles(&self) -> bool {
        // Use Kahn's algorithm to check for cycles
        let mut in_degree = self.in_degree.clone();
        let mut queue: Vec<TxIndex> = in_degree
            .iter()
            .enumerate()
            .filter_map(|(idx, &deg)| if deg == 0 { Some(idx) } else { None })
            .collect();

        let mut processed = 0;

        while let Some(tx_idx) = queue.pop() {
            processed += 1;
            for &dependent in &self.dependents[tx_idx] {
                in_degree[dependent] -= 1;
                if in_degree[dependent] == 0 {
                    queue.push(dependent);
                }
            }
        }

        processed != self.num_transactions
    }

    /// Get topological order of transactions
    pub fn topological_order(&self) -> Option<Vec<TxIndex>> {
        let mut result = Vec::with_capacity(self.num_transactions);
        let mut in_degree = self.in_degree.clone();
        let mut queue: Vec<TxIndex> = in_degree
            .iter()
            .enumerate()
            .filter_map(|(idx, &deg)| if deg == 0 { Some(idx) } else { None })
            .collect();

        while let Some(tx_idx) = queue.pop() {
            result.push(tx_idx);
            for &dependent in &self.dependents[tx_idx] {
                in_degree[dependent] -= 1;
                if in_degree[dependent] == 0 {
                    queue.push(dependent);
                }
            }
        }

        if result.len() == self.num_transactions {
            Some(result)
        } else {
            None // Cycle detected
        }
    }

    /// Get maximum parallelism level (size of largest independent set)
    pub fn max_parallelism(&self) -> usize {
        // Simple approximation: count transactions at each "level"
        let topo = match self.topological_order() {
            Some(order) => order,
            None => return 1, // Cycle, no parallelism
        };

        let mut levels = vec![0usize; self.num_transactions];
        let mut max_at_level = HashMap::new();

        for &tx_idx in &topo {
            let my_level = self.dependencies[tx_idx]
                .iter()
                .map(|&dep| levels[dep] + 1)
                .max()
                .unwrap_or(0);
            levels[tx_idx] = my_level;
            *max_at_level.entry(my_level).or_insert(0) += 1;
        }

        max_at_level.values().copied().max().unwrap_or(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pubkey(seed: u8) -> Pubkey {
        Pubkey([seed; 32])
    }

    #[test]
    fn test_no_conflicts_all_parallel() {
        // 3 transactions accessing different accounts
        let access_sets: Vec<AccessSet> = (0..3)
            .map(|i| {
                let mut a = AccessSet::new();
                a.add_write(make_pubkey(i));
                a
            })
            .collect();

        let graph = ConflictGraph::from_access_sets(&access_sets);

        // All should be ready (no dependencies)
        assert_eq!(graph.get_ready_transactions().len(), 3);
        assert_eq!(graph.max_parallelism(), 3);
    }

    #[test]
    fn test_chain_dependencies() {
        // 3 transactions all writing to same account = sequential chain
        let pk = make_pubkey(1);
        let access_sets: Vec<AccessSet> = (0..3)
            .map(|_| {
                let mut a = AccessSet::new();
                a.add_write(pk);
                a
            })
            .collect();

        let graph = ConflictGraph::from_access_sets(&access_sets);

        // Only first should be ready
        let ready = graph.get_ready_transactions();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0], 0);

        // Max parallelism should be 1 (all sequential)
        assert_eq!(graph.max_parallelism(), 1);
    }

    #[test]
    fn test_write_read_dependency() {
        let pk = make_pubkey(1);

        let mut access1 = AccessSet::new();
        access1.add_write(pk);

        let mut access2 = AccessSet::new();
        access2.add_read(pk);

        let graph = ConflictGraph::from_access_sets(&[access1, access2]);

        // tx 0 should be ready, tx 1 should wait
        assert!(graph.is_ready(0));
        assert!(!graph.is_ready(1));
        assert!(graph.get_dependencies(1).contains(&0));
    }

    #[test]
    fn test_no_cycles() {
        let access_sets: Vec<AccessSet> = (0..5)
            .map(|i| {
                let mut a = AccessSet::new();
                a.add_write(make_pubkey(i % 2)); // Interleaved conflicts
                a
            })
            .collect();

        let graph = ConflictGraph::from_access_sets(&access_sets);
        assert!(!graph.has_cycles());
    }
}
