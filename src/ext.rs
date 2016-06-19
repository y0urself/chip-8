//! Contains `extern "C" fn`s implementing higher-level opcodes.
//!
//! The JIT will generate calls to these methods to implement some opcodes.
//!
//! NOTE: None of these methods may unwind, since they would immediately unwind into JIT-compiled
//! code, which is undefined behaviour. We solve that problem by using the `UnwindAborter`, which
//! will abort the process in its `drop` method if the thread is unwinding.

use chip8::ChipState;

use std::thread::panicking;
use std::process::exit;

struct UnwindAborter;

impl Drop for UnwindAborter {
    fn drop(&mut self) {
        if panicking() {
            error!("aborting due to panic");
            exit(101);
        }
    }
}

/// Implements the `RND` CHIP-8 instruction.
///
/// `RND Vx, kk` generates a random byte, ANDs it with `kk`, and stores the result in `Vx`.
pub extern "C" fn rand(state: *mut ChipState, x: u8, kk: u8) {
    let _guard = UnwindAborter;
    unimplemented!();
}

/// Implements the `DRW` instruction.
///
/// `DRW Vx, Vy, n` draws an `n`-Byte sprite located at memory location `I` at x/y coordinates `Vx`
/// and `Vy`, and sets `VF` to 1 if a previously set pixel was unset, and to 0 if not (collision).
pub extern "C" fn draw(state: *mut ChipState, x: u8, y: u8, n: u8) {
    let _guard = UnwindAborter;
    unimplemented!();
}
