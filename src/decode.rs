// XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
// not the worst idea, but probably too much for this


//! A CHIP-8 instruction decoder.
//!
//! The decoder can theoretically be used to implement both an interpreter and a JIT.

/// Refers to the value of a general-purpose register or an immediate 8-bit value.
pub enum Value {
    Reg(u8),
    Imm(u8),
}

/// An operand for an abstract instruction.
pub enum Operand {
    /// A register or immediate.
    Value(Value),
    /// Add two values, evaluates to the result.
    Add(Value, Value),
    /// Add two values and, as a **side-effect**, set register `VF` to 1 on carry or to 0 if no
    /// carry occurs.
    AddWithCarry(Value, Value),
}

/// Trait to be implemented by users of the instruction decoder.
///
/// Defines methods called when encountering various kinds of instructions, as well as methods
/// needed by the decoder itself.
///
/// This interface operates on a more abstract version of the CHIP-8 ISA, which should be easier to
/// understand and implement.
pub trait Consumer {
    /// Called when the decoder wants to read the next instruction.
    ///
    /// This should return the next instruction word in host byte order.
    fn read_instruction(&mut self) -> u16;

    /// Evaluate an `Operand` and store the 8-bit result in `target_reg`.
    fn assign(&mut self, state: &mut DecoderState, target_reg: u8, operand: Operand);
}

/// Decoder state, passed to all consumer callbacks.
pub struct DecoderState {}

/// Instruction decoder.
pub struct Decoder<C: Consumer> {
    /// Current state, modified when decoding any instruction.
    state: DecoderState,
    /// The "consumer" using this decoder.
    consumer: C,
}

impl<C: Consumer> Decoder<C> {
    /// Create a new decoder that will start decoding at the given program counter value and call
    /// methods of `consumer` when an instruction was decoded.
    pub fn new(consumer: C) -> Self {
        Decoder {
            state: DecoderState {},
            consumer: consumer,
        }
    }

    /// Decodes a single basic block of CHIP-8 instructions, consuming the `Decoder`.
    pub fn decode_block(mut self) {
        loop {
            let instr = self.consumer.read_instruction();

            match instr {
                _ => unimplemented!(),
            }
        }
    }
}
