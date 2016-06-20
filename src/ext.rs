//! Contains `extern "C" fn`s implementing higher-level opcodes.
//!
//! The JIT will generate calls to these methods to implement some opcodes.
//!
//! FIXME: None of these methods may unwind, since they would immediately unwind into JIT-compiled
//! code, which is undefined behaviour. We should find an easy way to guard against that and abort
//! the process.

use chip8::ChipState;

use std::process::exit;

/// Implements the `RND` CHIP-8 instruction.
///
/// `RND Vx, kk` generates a random byte, ANDs it with `kk`, and stores the result in `Vx`.
#[allow(unused)]    // FIXME
pub unsafe extern "C" fn rand(state: *mut ChipState, x: u8, kk: u8) {
    println!("rand unimplemented");
    exit(102);
}

/// Implements the `SKP Vx` and `SKNP Vx` instructions.
///
/// Returns `true` (`1`) when the key stored in `Vx` is pressed, `false` (`0`) if not.
pub unsafe extern "C" fn key_pressed(_state: *mut ChipState, _x: u8) -> bool {
    warn!("key_pressed unimplemented");
    false
}

/// Implements the `LD Vx, [I]` instruction.
///
/// Reads registers `V0` through `Vx` from memory starting at location `I`.
pub unsafe extern "C" fn load_mem(state: *mut ChipState, x: u8) {
    let state = &mut *state;
    for idx in 0..x+1 {
        state.regs[idx as usize] = state.mem[state.i as usize + idx as usize];
    }
}

/// Implements the `DRW` instruction.
///
/// `DRW Vx, Vy, n` draws an `n`-Byte sprite located at memory location `I` at x/y coordinates `Vx`
/// and `Vy`, and sets `VF` to 1 if a previously set pixel was unset, and to 0 if not (collision).
pub unsafe extern "C" fn draw(state: *mut ChipState, x: u8, y: u8, n: u8) {
    debug!("{:?}: DRW V{:01X}, V{:01X}, {}", state, x, y, n);

    // TODO
}
