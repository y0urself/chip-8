
// This is pretty oversimplified and only handles functions that take a few integer arguments (so
// that we won't use the stack to pass any), no float arguments, and return nothing.

use chip8::ChipState;

/// Trait for functions the JIT can generate calls for.
pub trait Callable {
    fn param_count() -> u8;
    fn get_addr(&self) -> u64;
}

impl Callable for unsafe extern "C" fn(*mut ChipState, u8, u8, u8) {
     fn param_count() -> u8 { 4 }
     fn get_addr(&self) -> u64 { *self as u64 }
}

impl Callable for unsafe extern "C" fn(*mut ChipState, u8, u8) {
     fn param_count() -> u8 { 3 }
     fn get_addr(&self) -> u64 { *self as u64 }
}

impl<T> Callable for unsafe extern "C" fn(*mut ChipState, u8) -> T {
     fn param_count() -> u8 { 2 }
     fn get_addr(&self) -> u64 { *self as u64 }
}

impl Callable for unsafe extern "C" fn(*mut ChipState) {
    fn param_count() -> u8 { 1 }
    fn get_addr(&self) -> u64 { *self as u64 }
}

/// Returns the registers in which arguments are passed to functions, as their encoding to specify
/// in the instruction.
pub fn get_arg_reg_codes() -> &'static [u8] {
    const RDI: u8 = 0b111;
    const RSI: u8 = 0b110;
    const RDX: u8 = 0b010;
    const RCX: u8 = 0b001;

    if cfg!(all(target_family = "unix", target_arch = "x86_64")) {
        // "The first six integer or pointer arguments are passed in registers RDI, RSI, RDX, RCX
        // (R10 in the Linux kernel interface), R8, and R9."
        // For us, 4 registers are currently enough. Using R8 and R9 would require using a REX
        // prefix, which I don't want to do right now.
        static RESULT: &'static [u8] = &[RDI, RSI, RDX, RCX];
        RESULT
    } else if cfg!(all(target_family = "windows", target_arch = "x86_64")) {
        // XXX Keep in mind that the Microsoft ABI needs 32 bytes of shadow space.
        unimplemented!();
    } else {
        panic!("unsupported host arch :(");
    }
}
