//! Contains `extern "C" fn`s implementing higher-level opcodes.
//!
//! The JIT will generate calls to these methods to implement some opcodes.
//!
//! FIXME: None of these methods may unwind, since they would immediately unwind into JIT-compiled
//! code, which is undefined behaviour. We should find an easy way to guard against that and abort
//! the process.

use chip8::ChipState;

use rand::random;
use minifb::Key;    // XXX <- This enum is undocumented, I assume by accident?

fn hexkey_to_key(key: u8) -> Key {
    match key {
        0x0 => Key::Key0,
        0x1 => Key::Key1,
        0x2 => Key::Key2,
        0x3 => Key::Key3,
        0x4 => Key::Key4,
        0x5 => Key::Key5,
        0x6 => Key::Key6,
        0x7 => Key::Key7,
        0x8 => Key::Key8,
        0x9 => Key::Key9,
        0xA => Key::A,
        0xB => Key::B,
        0xC => Key::C,
        0xD => Key::D,
        0xE => Key::E,
        0xF => Key::F,
        _ => panic!("invalid key {:01X}", key),
    }
}

/// Implements the `RND` CHIP-8 instruction.
///
/// `RND Vx, kk` generates a random byte, ANDs it with `kk`, and stores the result in `Vx`.
pub unsafe extern "C" fn rand(state: *mut ChipState, x: u8, kk: u8) {
    let state = &mut *state;
    state.regs[x as usize] = random::<u8>() & kk;
}

/// Implements the `SKP Vx` and `SKNP Vx` instructions.
///
/// Returns `true` (`1`) when the key stored in `Vx` is pressed, `false` (`0`) if not.
pub unsafe extern "C" fn key_pressed(state: *mut ChipState, x: u8) -> bool {
    let state = &mut *state;
    let key = state.regs[x as usize] & 0xF;

    state.win.is_key_down(hexkey_to_key(key))
}

/// Implements the `LD Vx, K` instruction.
pub unsafe extern "C" fn wait_key_press(state: *mut ChipState, x: u8) {
    let state = &mut *state;
    loop {
        state.win.update();

        if !state.win.is_open() {
            // The program will exit on the next tick.
            // XXX Note that this isn't really correct since we don't store a value in the reg!
            break;
        }

        for k in 0..16 {
            if state.win.is_key_down(hexkey_to_key(k)) {
                state.regs[x as usize] = k;
                return;
            }
        }
    }
}

/// Implements the `LD Vx, [I]` instruction.
///
/// Reads registers `V0` through `Vx` from memory starting at location `I`.
pub unsafe extern "C" fn load_mem(state: *mut ChipState, x: u8) {
    let state = &mut *state;
    for idx in 0..x+1 {
        state.regs[idx as usize] = state.mem[state.i as usize + idx as usize];
    }

    // "I is set to I + X + 1 after operation"
    state.i = state.i + x as u16 + 1;
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

    state.inv_start = state.i;
    state.inv_len = x + 1;

    // "I is set to I + X + 1 after operation"
    state.i = state.i + x as u16 + 1;
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

    state.inv_start = state.i;
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
    let state = &mut *state;
    let x = state.regs[x as usize] as usize;
    let y = state.regs[y as usize] as usize;

    let mut cleared = false;
    for i in 0..n as usize {
        // Draw 8-pixel sprite line to `x..x+8`
        let y_total = (y + i) % 32;

        let data = state.mem[state.i as usize + i as usize];
        for xoff in 0..8 {
            let x_total = (x + xoff) % 64;

            let mut pixref = &mut state.fb[y_total * 64 + x_total];
            let newval = data & (0x80 >> xoff) != 0;
            if *pixref && newval {
                // Set pixel will be cleared
                cleared = true;
            }
            *pixref ^= newval;
        }
    }

    state.regs[0xF] = if cleared { 1 } else { 0 };
}

/// Implements `CLS`.
pub unsafe extern "C" fn clear_screen(state: *mut ChipState) {
    let fb = &mut (*state).fb[..];
    for pixel in fb {
        *pixel = false;
    }
}
