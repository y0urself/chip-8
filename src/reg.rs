//! Defines X86 registers

use enum_set::CLike;

use std::mem;

/// List of x86 registers that can be allocated to hold values of CHIP-8 registers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum X86Register {
    AH,
    AL,
    BH,
    BL,
    CH,
    CL,
    DH,
    DL,
}

/// An iterator over the variants of `X86Register`.
pub struct Iter(u8);

impl Iterator for Iter {
    type Item = X86Register;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0 > X86Register::DL as u8 {  // XXX must be the last variant
            None
        } else {
            let variant = self.0 as u32;
            self.0 += 1;
            unsafe { Some(X86Register::from_u32(variant)) }
        }
    }
}

impl X86Register {
    pub fn iter() -> Iter {
        Iter(0)
    }

    /// Get the value for encoding this 8-bit register in a `ModR/M` byte's `REG` field.
    ///
    /// Note that this is only correct for operations with an 8-bit data size. Bigger sizes use
    /// bigger registers, obviously.
    pub fn as_reg_field_value(&self) -> u8 {
        match *self {
            X86Register::AH => 0b100,
            X86Register::AL => 0b000,
            X86Register::BH => 0b111,
            X86Register::BL => 0b011,
            X86Register::CH => 0b101,
            X86Register::CL => 0b001,
            X86Register::DH => 0b110,
            X86Register::DL => 0b010,
        }
    }

    /// Similar to `as_reg_field_value`, but gets the encoding of the 16/32-bit register
    /// *containing* `self`.
    pub fn as_r32_value(&self) -> u8 {
        match *self {
            X86Register::AH | X86Register::AL => 0b000, // (e)ax
            X86Register::BH | X86Register::BL => 0b011, // (e)bx
            X86Register::CH | X86Register::CL => 0b001, // (e)cx
            X86Register::DH | X86Register::DL => 0b010, // (e)dx
        }
    }

    /// Gets the other half of this register.
    pub fn other_half(&self) -> Self {
        match *self {
            X86Register::AH => X86Register::AL,
            X86Register::AL => X86Register::AH,
            X86Register::BH => X86Register::BL,
            X86Register::BL => X86Register::BH,
            X86Register::CH => X86Register::CL,
            X86Register::CL => X86Register::CH,
            X86Register::DH => X86Register::DL,
            X86Register::DL => X86Register::DH,
        }
    }

    /// Is this (part of) a callee-saved register? If yes, we need to generate code that does
    /// exactly that.
    ///
    /// If a register is not callee-saved, it is caller-saved, which means that we must save its
    /// value before calling an external function.
    ///
    /// This is part of the host's calling convention.
    pub fn is_callee_saved(&self) -> bool {
        if cfg!(all(target_family = "unix", target_arch = "x86_64")) {
            // In the System V x64 ABI, `rbx`, `rbp` and `r12-r15` are callee-saved.
            match *self {
                X86Register::BH | X86Register::BL => true,
                _ => false,
            }
        } else if cfg!(all(target_family = "windows", target_arch = "x86_64")) {
            // "The registers RBX, RBP, RDI, RSI, RSP, R12, R13, R14, and R15 are considered
            // nonvolatile and must be saved and restored by a function that uses them."
            match *self {
                X86Register::BH | X86Register::BL => true,
                _ => false,
            }
        } else {
            panic!("unsupported host arch :(");
        }
    }
}

impl CLike for X86Register {
    fn to_u32(&self) -> u32 {
        *self as u32
    }

    unsafe fn from_u32(val: u32) -> Self {
        mem::transmute(val as u8)
    }
}
