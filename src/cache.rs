//! JIT code cache.
//!
//! Provides a helper function to run JITted from the current program counter. If the block wasn't
//! yet translated, the JIT compiler is invoked.
//!
//! Any time CHIP-8 code is executed, it can modify RAM contents using the `LD [I], Vx` or
//! `LD B, Vx` instructions, so the cache must be kept consistent (code overwritten by the program
//! must be recompiled).

use jit::{self, CompiledBlock};
use chip8::ChipState;

use fnv::FnvHasher;

use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::rc::Rc;
use std::cell::RefCell;

type FnvHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FnvHasher>>;

pub struct CodeCache {
    block_map: FnvHashMap<u16, CompiledBlock>,
}

impl CodeCache {
    pub fn new() -> Self {
        CodeCache {
            block_map: FnvHashMap::with_hasher(Default::default()),
        }
    }

    /// Gets or compiles the code block starting at the current program counter.
    pub fn get_or_compile_block(&mut self, state_rc: &Rc<RefCell<ChipState>>) -> &CompiledBlock {
        let pc = state_rc.borrow().pc;
        self.block_map.entry(pc).or_insert_with(|| {
            debug!("cache miss for pc ${:04X}", pc);
            jit::compile(state_rc.clone())
        })
    }

    #[allow(dead_code)] // FIXME
    pub fn invalidate_range(&mut self, start: u16, len: u16) {
        debug!("invalidating range {:03X}..{:03X}", start, start + len);

        let end = start + len;
        // FIXME: This is slow, since it iterates over the whole cache and recreates the whole map.
        let mut newmap = FnvHashMap::with_hasher(Default::default());
        for (pc, block) in self.block_map.drain() {
            if block.start > end || block.start + block.len < start {
                // no overlap
                newmap.insert(pc, block);
            } else {
                debug!("invalidated block at {:03X}..{:03X}", block.start, block.start + block.len);
            }
        }

        self.block_map = newmap;
    }
}
