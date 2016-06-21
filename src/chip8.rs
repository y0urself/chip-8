use cache::CodeCache;

use minifb::{Window, WindowOptions, Scale};

use std::rc::Rc;
use std::cell::RefCell;
use std::process;
use std::time::{Instant, Duration};
use std::thread::sleep;

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
    /// CHIP-8 stack, containing up to 16 return addresses.
    pub stack: [u16; 16],
    // Pointer into `stack`. The starting value doesn't matter, since `CALL` and `RET` will mask
    // this with `0b1111`.
    pub sp: u8,

    /// Current delay timer value, decremented at 60 Hz when non-zero.
    pub delay_timer: u8,
    /// Current sound timer value, decremented at 60 Hz when non-zero. While non-zero, a sound is
    /// played.
    pub sound_timer: u8,

    /// Frame buffer. `true` = "On", `false` = "Off" (initial value).
    pub fb: [bool; 64 * 32],

    /// Set by JIT-compiled code when we need to invalidate `inv_len` bytes of `mem` starting at
    /// `i`. This must be checked after every block is executed.
    pub inv_len: u8,

    pub win: Window,

    _priv: (),
}

pub struct Chip8 {
    state: Rc<RefCell<ChipState>>,
    cache: CodeCache,
    last_tick: Instant,
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
                stack: [0; 16],
                sp: 0,
                delay_timer: 0,
                sound_timer: 0,
                fb: [false; 64 * 32],
                inv_len: 0,
                win: Window::new("KOOL-8", 64, 32, WindowOptions {
                    borderless: false,
                    title: true,
                    resize: false,
                    scale: Scale::FitScreen,
                }).unwrap(),
                _priv: (),
            })),
            cache: CodeCache::new(),
            last_tick: Instant::now(),
        }
    }

    /// Called in intervals by the main loop
    fn tick(&mut self) {
        let mut state = self.state.borrow_mut();

        if !state.win.is_open() {
            info!("window closed, exiting");
            process::exit(0);
        }

        // Render the screen
        let mut screen_buf = [0u32; 64 * 32];
        for (i, &pix) in state.fb.iter().enumerate() {
            if pix {
                screen_buf[i] = 0xffffffff;
            }
        }
        state.win.update_with_buffer(&screen_buf);

        // Decrease counters
        if state.delay_timer > 0 {
            state.delay_timer -= 1;
        }
        if state.sound_timer > 0 {
            state.sound_timer -= 1;
        }

        // Sleep so we tick at 60 Hz
        let left_time = Duration::from_millis(1000 / 60) - self.last_tick.elapsed();
        sleep(left_time);

        self.last_tick = Instant::now();
    }

    /// Begins execution of CHIP-8 instructions.
    pub fn run(&mut self) {
        // We count the block we execute, and after we've run "enough", we'll do other tasks
        let mut block_counter = 0;
        const BLOCKS_PER_TICK: u32 = 4096;

        loop {
            trace!("@ {:04X}", self.state.borrow().pc);

            {
                let block = self.cache.get_or_compile_block(&self.state);
                block.run();
            }

            {
                let mut state = self.state.borrow_mut();
                let inv_start = state.i;
                let inv_len = &mut state.inv_len;
                if *inv_len != 0 {
                    // JIT cache invalidation required
                    self.cache.invalidate_range(inv_start, *inv_len as u16);
                    *inv_len = 0;
                }
            }

            block_counter += 1;

            if block_counter >= BLOCKS_PER_TICK {
                block_counter = 0;
                self.tick();
            }
        }
    }
}
