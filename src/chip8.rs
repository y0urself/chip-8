use cache::CodeCache;

use std::rc::Rc;
use std::cell::RefCell;

static CHIP8_FONT: [u8; 5 * 16] = [
    0xf0, 0x90, 0x90, 0x90, 0xf0,   // 0
    0x20, 0x60, 0x20, 0x20, 0x70,   // 1
    0xf0, 0x10, 0xf0, 0x80, 0xf0,   // 2
    0xf0, 0x10, 0xf0, 0x10, 0xf0,   // 3
    0xf0, 0x80, 0xf0, 0x90, 0xf0,   // 4
    0xf0, 0x80, 0xf0, 0x10, 0xf0,   // 5
    0xf0, 0x80, 0xf0, 0x90, 0xf0,   // 6
    0xf0, 0x10, 0x20, 0x40, 0x40,   // 7
    0xf0, 0x90, 0xf0, 0x90, 0xf0,   // 8
    0xf0, 0x90, 0xf0, 0x10, 0xf0,   // 9
    0xf0, 0x90, 0xf0, 0x90, 0x90,   // A
    0xe0, 0xe0, 0xe0, 0x90, 0xe0,   // B
    0xf0, 0x80, 0x80, 0x80, 0xf0,   // C
    0xe0, 0x90, 0x90, 0x90, 0xe0,   // D
    0xf0, 0x80, 0xf0, 0x80, 0xf0,   // E
    0xf0, 0x80, 0xf0, 0x80, 0x80,   // F
];

const RAM_SIZE: usize = 4096;
/// Program start address in RAM
const PROGRAM_START: usize = 0x200;

/// CHIP-8 state, also accessed by JITted code, so must always have a fixed address.
#[repr(C)]
pub struct ChipState {
    /// CHIP-8 memory.
    ///
    /// The ROM is copied to address `0x200` in here, the low 512 bytes can be freely used by the
    /// interpreter (`0x000 - `0x1ff`). We only store the font there.
    pub mem: [u8; RAM_SIZE],
    /// General-purpose registers (`Vx`).
    pub regs: [u8; 16],
    /// Program counter. Contains the address of the next instruction.
    pub pc: u16,
    /// The index register I.
    pub i: u16,

    /// Current delay timer value, decremented at 60 Hz when non-zero.
    pub delay_timer: u8,
    /// Current sound timer value, decremented at 60 Hz when non-zero. While non-zero, a sound is
    /// played.
    pub sound_timer: u8,

    _priv: (),
}

pub struct Chip8 {
    state: Rc<RefCell<ChipState>>,
    cache: CodeCache,
}

impl Chip8 {
    /// Creates a new CHIP-8 emulator loading the given ROM. The ROM must not be larger than 3584
    /// Bytes.
    pub fn new(rom: &[u8]) -> Self {
        if rom.len() > RAM_SIZE - PROGRAM_START {
            panic!("ROM is too large to fit in RAM");
        }

        let mut mem = [0; RAM_SIZE];
        for (i, &b) in CHIP8_FONT.iter().enumerate() { mem[i] = b; }
        for (i, &b) in rom.iter().enumerate() { mem[PROGRAM_START + i] = b; }

        Chip8 {
            state: Rc::new(RefCell::new(ChipState {
                mem: mem,
                regs: [0; 16],
                pc: PROGRAM_START as u16,
                i: 0,
                delay_timer: 0,
                sound_timer: 0,
                _priv: (),
            })),
            cache: CodeCache::new(),
        }
    }

    /// Begins execution of CHIP-8 instructions.
    pub fn run(&mut self) {
        loop {
            trace!("@ {:04X}", self.state.borrow().pc);
            let block = self.cache.get_or_compile_block(&self.state);
            block.run();
        }
    }
}
