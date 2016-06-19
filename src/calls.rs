//! Calling convention logic for the JIT

use chip8::ChipState;

/// Trait for functions the JIT can generate calls for.
pub trait Callable {
    
}

impl Callable for extern "C" fn(*mut ChipState) {

}
