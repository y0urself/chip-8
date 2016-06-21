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

/// Implements the `LD Vx, K` instruction.
pub unsafe extern "C" fn wait_key_press(_state: *mut ChipState, _x: u8) {
    warn!("wait_key_press unimplemented, entering infinite loop");
    loop {}
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

/// Implements the `LD [I], Vx` instruction.
///
/// Stores registers `V0` through `Vx` to memory starting at location `I`.
///
/// Since this kind of write can overwrite machine code (not that it *should* do that), we need to
/// invalidate JIT-compiled code from that RAM section.
pub unsafe extern "C" fn store_mem(state: *mut ChipState, x: u8) {
    let state = &mut *state;
    for idx in 0..x+1 {
        state.mem[state.i as usize + idx as usize] = state.regs[idx as usize];
    }

    state.inv_len = x + 1;
}

/// Implements the `LD B, Vx` instruction.
///
/// Stores the BCD representation of `Vx` in memory locations `I`, `I+1`, and `I+2`.
///
/// Since this also writes to memory, we need to invalidate the JIT cache.
pub unsafe extern "C" fn binary_to_bcd(state: *mut ChipState, x: u8) {
    let state = &mut *state;
    let value = state.regs[x as usize];
    state.mem[state.i as usize] = value / 100;
    state.mem[state.i as usize + 1] = (value % 100) / 10;
    state.mem[state.i as usize + 2] = value % 10;

    state.inv_len = 3;
}

/// Implements the `LD F, Vx` instruction.
///
/// Sets I = location of sprite for digit Vx.
pub unsafe extern "C" fn hex_sprite_address(state: *mut ChipState, x: u8) {
    let state = &mut *state;
    let value = state.regs[x as usize];

    // 5 Bytes per sprite
    state.i = (value as u16 & 0xff) * 5;
}

/// Implements the `DRW` instruction.
///
/// `DRW Vx, Vy, n` draws an `n`-Byte sprite located at memory location `I` at x/y coordinates `Vx`
/// and `Vy`, and sets `VF` to 1 if a previously set pixel was unset, and to 0 if not (collision).
pub unsafe extern "C" fn draw(state: *mut ChipState, x: u8, y: u8, n: u8) {
    debug!("{:?}: DRW V{:01X}, V{:01X}, {}", state, x, y, n);

    // TODO
}

/// Implements `CLS`.
pub unsafe extern "C" fn clear_screen(state: *mut ChipState) {
    let fb = &mut (*state).fb[..];
    for pixel in fb {
        *pixel = false;
    }
}
