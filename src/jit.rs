//! Implements the x86-64 just-in-time compiler / "dynarec".
//!
//! The JIT compiles a single basic block at a time, resulting in one function per block. The
//! emulator will run a loop which compiles the block starting at the current program counter, calls
//! the resulting function, and repeats. This is quite inefficient, but it's common practice, so I
//! won't do anything more fancy than that for now.
//!
//! The generated code will always store a reference to the `ChipState` in the `rsi` register, so it
//! can easily access any field inside it. Thus, one of the first things a compiled function does is
//! loading the address of the state into `rsi.`

use calls::{self, Callable};
use chip8::ChipState;
use reg::X86Register;
use hexdump::hexdump;
use ext;

use byteorder::{LittleEndian, BigEndian, WriteBytesExt, ReadBytesExt};
use memmap::{Mmap, Protection};
use enum_set::EnumSet;

use std::mem;
use std::rc::Rc;
use std::cell::{Ref, RefCell};
use std::i32;
use std::marker::PhantomData;

/// A compiled block of machine code, created by the JIT.
pub struct CompiledBlock {
    /// Compiled code will access (and mutate!) the `ChipState` passed to the JIT at creation.
    state: Rc<RefCell<ChipState>>,
    /// The memory allocation backing the code of this block
    memory: Mmap,
    /// Start address in CHIP-8 RAM.
    pub start: u16,
    /// Length of the CHIP-8 code from which this block was compiled.
    pub len: u16,
}

impl CompiledBlock {
    /// Executes this block of code.
    pub fn run(&self) {
        // The code will modify the `ChipState`, so we need to hold a mutable borrow as long as the
        // code runs.
        let _state = self.state.borrow_mut();

        let fnptr: unsafe extern "C" fn() = unsafe { mem::transmute(self.memory.ptr()) };

        unsafe {
            fnptr();
        }
    }
}

/// Field offset newtype, for type safety. Stores the offset in bytes from the beginning of the
/// `ChipState`.
struct Offset<T> {
    bytes: usize,
    _p: PhantomData<T>,
}

/// Operations that can be performed on a register and an immediate 8-bit value.
enum ImmediateOp {
    Add = 0,
    Or = 1,
    Adc = 2,
    Sbb = 3,
    And = 4,
    Sub = 5,
    Xor = 6,
    Cmp = 7,
}

/// JIT compiler entry point.
///
/// This will compile the basic block starting at the current program counter, and return it as a
/// `CompiledBlock`.
pub fn compile(state_rc: Rc<RefCell<ChipState>>) -> CompiledBlock {
    debug!("jitting block at ${:04X}", state_rc.borrow().pc);

    let jit = Jit::new(state_rc);
    jit.compile()
}

struct Jit {
    /// Program counter at which we started to compile the block.
    start_pc: u16,
    /// JIT-internal program counter pointing at the next instruction we'll *compile*, not execute.
    pc: u16,
    /// Reference to the `ChipState`. Used for generating code that accesses the emulator state.
    state_rc: Rc<RefCell<ChipState>>,
    /// Buffer for generated machine code. Filled while the JIT is doing its thing, later copied to
    /// an executable memory region.
    code_buffer: Vec<u8>,

    /// Register allocation map. Maps CHIP-8 register indices to host registers containing their
    /// values in this block, and a `u16` counting how often this CHIP-8 register has been used
    /// since it was allocated (used for spill weight calculation).
    reg_map: [Option<(X86Register, u16)>; 16],
    /// Contains all host registers available for allocation without spilling.
    free_host_regs: EnumSet<X86Register>,
    /// Current use index. Incremented each time a CHIP-8 register is used.
    use_index: u16,
    /// When we allocate a callee-saved host register, we push it to the stack to save its old
    /// value. We also push the register to this vector. When generating code to leave the JITted
    /// function, we need to pop them back off. That's done by iterating through this vector in
    /// reverse order.
    ///
    /// Since host regs only need to be saved once, this should contain each host reg at most once.
    saved_host_regs: Vec<X86Register>,
}

impl Jit {
    /// Creates a new JIT context
    fn new(state_rc: Rc<RefCell<ChipState>>) -> Jit {
        let pc = state_rc.borrow().pc;
        Jit {
            start_pc: pc,
            pc: pc,
            state_rc: state_rc,
            code_buffer: Vec::new(),
            reg_map: [None; 16],
            free_host_regs: X86Register::iter().collect(),
            use_index: 0,
            saved_host_regs: Vec::new(),
        }
    }

    /// Marks all host regs as free
    fn reset_free_host_regs(&mut self) {
        self.free_host_regs = X86Register::iter().collect();
    }

    /// Compiles the given CHIP-8 instruction.
    ///
    /// Returns `true` if the block was finished and compilation is done.
    fn compile_instr(&mut self, instr: u16) -> bool {
        // FIXME Maybe replace the `bool` with something more type safe?
        trace!("compile_instr ${:04X}", instr);

        let nibbles = [
            (instr & 0xf000) >> 12,
            (instr & 0x0f00) >> 8,
            (instr & 0x00f0) >> 4,
            (instr & 0x000f)
        ];

        // me want slice patterns
        match (nibbles[0], nibbles[1], nibbles[2], nibbles[3]) {
            (0x0, 0x0, 0xE, 0x0) => {
                trace!("-> CLS");
                unimplemented!();
            }
            (0x0, 0x0, 0xE, 0xE) => {
                trace!("-> RET");
                unimplemented!();
            }
            (0x1, _, _, _) => {
                let addr = instr & 0x0fff;
                trace!("-> JP ${:03X}", addr);

                let pc_offset = self.calc_offset(|state| &state.pc);
                self.emit_state_store_u16(pc_offset, addr);
                true
            }
            (0x2, _, _, _) => {
                let addr = instr & 0x0fff;
                trace!("-> CALL {:03X}", addr);
                unimplemented!();
            }
            (0x3, x, _, _) => {
                // Skip next instruction if `Vx == kk`.
                // Load new program counter with `self.pc` or `self.pc + 2` depending on `Vx == kk`.
                let k = instr & 0xff;
                trace!("-> SE V{:01X}, {:02X}", x, k);

                let reg = self.get_host_reg_for(x as u8);

                // We'll use `di` as a scratch register
                // `mov di, <PC>`
                self.emit_raw(&[0x66, 0xBF]);
                self.code_buffer.write_u16::<LittleEndian>(self.pc);

                // `cmp <REG>, <VAL>`
                self.emit_raw(&[0x80, 0xF8 | reg.as_reg_field_value(), k as u8]);

                // Skip the increment if not equal
                // `jne +<DIST>`
                let dist = 6;
                self.emit_raw(&[0x0F, 0x85]);
                self.code_buffer.write_i32::<LittleEndian>(dist);

                // Increment PC by 2
                // Just do `inc di` twice
                self.emit_raw(&[0x66, 0xFF, 0xC7]);
                self.emit_raw(&[0x66, 0xFF, 0xC7]);

                // Store PC in `ChipState`
                let offset = self.calc_offset(|state| &state.pc);
                // `mov [rsi + <OFFSET>], di`
                self.emit_raw(&[0x66, 0x89, 0xBE]);
                // FIXME check offset validity
                self.code_buffer.write_i32::<LittleEndian>(offset.bytes as i32);

                true
            }
            (0x4, x, _, _) => {
                let k = instr & 0xff;
                trace!("-> SNE V{:01X}, {:02X}", x, k);
                unimplemented!();
            }
            (0x5, x, y, 0x0) => {
                trace!("-> SE V{:01X}, V{:01X}", x, y);
                unimplemented!();
            }
            (0x6, x, _, _) => {
                let k = instr & 0xff;
                trace!("-> LD V{:01X}, {:02X}", x, k);

                // Load the immediate into the allocated host register
                let host_reg = self.get_host_reg_for(x as u8);
                self.emit_load_imm8(host_reg, k as u8);
                false
            }
            (0x7, x, _, _) => {
                let k = instr & 0xff;
                trace!("-> ADD V{:01X}, {:02X}", x, k);

                let reg = self.get_host_reg_for(x as u8);
                self.emit_imm_op(reg, ImmediateOp::Add, k as u8);
                false
            }
            (0x8, x, y, 0x0) => {
                trace!("-> LD V{:01X}, V{:01X}", x, y);
                unimplemented!();
            }
            (0x8, x, y, 0x1) => {
                trace!("-> OR V{:01X}, V{:01X}", x, y);
                unimplemented!();
            }
            (0x8, x, y, 0x2) => {
                trace!("-> AND V{:01X}, V{:01X}", x, y);
                unimplemented!();
            }
            (0x8, x, y, 0x3) => {
                trace!("-> XOR V{:01X}, V{:01X}", x, y);
                unimplemented!();
            }
            (0x8, x, y, 0x4) => {   // VF = Carry (0 or 1)
                trace!("-> ADD V{:01X}, V{:01X}", x, y);
                unimplemented!();
            }
            (0x8, x, y, 0x5) => {   // VF = !Borrow
                trace!("-> SUB V{:01X}, V{:01X}", x, y);
                unimplemented!();
            }
            (0x8, x, y, 0x6) => {   // VF = LSb of Vx
                trace!("-> SHR V{:01X}   ; V{:01X}", x, y);
                unimplemented!();
            }
            (0x8, x, y, 0x7) => {   // VF = !Borrow
                trace!("-> SUBN V{:01X}, V{:01X}", x, y);
                unimplemented!();
            }
            (0x8, x, y, 0xE) => {   // VF = MSb of Vx
                trace!("-> SHL V{:01X}   ; V{:01X}", x, y);
                unimplemented!();
            }
            (0x9, x, y, 0x0) => {
                trace!("-> SNE V{:01X}, V{:01X}", x, y);
                unimplemented!();
            }
            (0xA, _, _, _) => {
                let nnn = instr & 0x0fff;
                trace!("-> LD I, {:03X}", nnn);

                // Currently, we don't allocate a host reg for I, so this just stores to the
                // `ChipState`.
                let offset = self.calc_offset(|state| &state.i);
                self.emit_state_store_u16(offset, nnn);
                false
            }
            (0xB, _, _, _) => {
                let nnn = instr & 0x0fff;
                trace!("-> JP V0, {:03X}", nnn);
                unimplemented!();
            }
            (0xC, x, _, _) => {
                let k = instr & 0x00ff;
                trace!("-> RND V{:01X}, {:02X}", x, k);
                unimplemented!();
            }
            (0xD, x, y, n) => {
                trace!("-> DRW V{:01X}, V{:01X}, {:01X}", x, y, n);

                let state = self.state_address();
                let fptr = ext::draw as extern "C" fn(_, _, _, _);  // XXX this is dumb
                self.emit_call(fptr, &[state as u64, x as u64, y as u64, n as u64]);
                false
            }
            (0xE, x, 0x9, 0xE) => {
                trace!("-> SKP V{:01X}", x);
                unimplemented!();
            }
            (0xE, x, 0xA, 0x1) => {
                trace!("-> SKNP V{:01X}", x);
                unimplemented!();
            }
            (0xF, x, 0x0, 0x7) => {
                trace!("-> LD V{:01X}, DT", x);
                unimplemented!();
            }
            (0xF, x, 0x0, 0xA) => {
                trace!("-> LD V{:01X}, K", x);
                unimplemented!();
            }
            (0xF, x, 0x1, 0x5) => {
                trace!("-> LD DT, V{:01X}", x);
                unimplemented!();
            }
            (0xF, x, 0x1, 0x8) => {
                trace!("-> LD ST, V{:01X}", x);
                unimplemented!();
            }
            (0xF, x, 0x1, 0xE) => {
                trace!("-> ADD I, V{:01X}", x);

                let reg = self.get_host_reg_for(x as u8);
                let offset = self.calc_offset(|state| &state.i);

                // 16-bit addition requires the register to be 16-bit too, so we move the reg value
                // to `di` which is otherwise unused by us. This means we need to save `di` in case
                // it's callee-saved!

                // `mov di, <REG>` doesn't work since the regs have different sizes, so we'll use
                // `movzx di, <REG>`.
                self.emit_raw(&[0x66, 0x0f, 0xb6, 0xf8 | reg.as_reg_field_value()]);

                // `add word ptr [rsi + <OFFSET>], di`
                self.emit_raw(&[0x66, 0x01, 0xbe]);
                self.code_buffer.write_i32::<LittleEndian>(offset.bytes as i32).unwrap();
                false
            }
            (0xF, x, 0x2, 0x9) => {
                trace!("-> LD F, V{:01X}", x);
                unimplemented!();
            }
            (0xF, x, 0x3, 0x3) => {
                trace!("-> LD B, V{:01X}", x);
                unimplemented!();
            }
            (0xF, x, 0x5, 0x5) => {
                trace!("-> LD [I], V{:01X}", x);
                unimplemented!();
            }
            (0xF, x, 0x6, 0x5) => {
                trace!("-> LD V{:01X}, [I]", x);
                unimplemented!();
            }
            _ => {
                warn!("ignoring unknown instruction ${:04X}", instr);
                false
            }
        }
    }

    /// Writes the function prolog, which makes sure all invariants held by the JIT are met after
    /// this has executed.
    fn emit_prolog(&mut self) {
        // `esi` is callee-saved at least on Windows.
        // FIXME: We should express this in a better way. `reg` contains callee-saved info, but it's
        // only used for allocable regs.
        self.emit_rsi_save();

        let state_addr = self.state_address() as *mut ChipState;
        self.emit_load_rsi_ptr(state_addr);
    }

    /// Finalizes a function's JIT code by synchronizing all CHIP-8 registers, restoring all callee-
    /// saved host registers and emitting a `ret`.
    fn finalize_function(&mut self) {
        for reg in 0..16 {
            self.spill_chip8_reg(reg);
        }

        self.emit_restore_regs();

        self.emit_rsi_restore();

        // `ret`
        self.emit_raw(&[0xC3]);
    }

    /// Compiles the basic block starting at the current program counter.
    fn compile(mut self) -> CompiledBlock {
        self.emit_prolog();

        // Copy out `mem` because borrowck complains. Luckily, the memory is quite small, but I
        // wonder if there isn't a better solution (that doesn't involve calling `borrow_mut` every
        // like 3 cycles).
        let mut mem = self.state_rc.borrow_mut().mem;
        loop {
            let instr = (&mem[self.pc as usize..]).read_u16::<BigEndian>().unwrap();
            self.pc += 2;
            if self.compile_instr(instr) {
                break;
            }
        }

        self.finalize_function();

        trace!("jit completed, code dump: {}", hexdump(&self.code_buffer));

        // Copy the code buffer into a memory region and make it executable:
        let mut mmap = Mmap::anonymous(self.code_buffer.len(), Protection::ReadWrite).unwrap();
        unsafe {
            mmap.as_mut_slice().copy_from_slice(&self.code_buffer);
        }

        mmap.set_protection(Protection::ReadExecute).unwrap();

        CompiledBlock {
            state: self.state_rc,
            memory: mmap,
            start: self.start_pc,
            len: self.pc - self.start_pc,
        }
    }

    /// Returns the x86 register holding `chip_reg`s value.
    ///
    /// This may emit code to fetch the value of the register from the emulator state, or to flush
    /// an allocated register to the emulator state.
    fn get_host_reg_for(&mut self, chip_reg: u8) -> X86Register {
        // We'll always return a good register here, this method cannot fail. Since every call means
        // a use, increment `use_index`.
        self.use_index += 1;

        if let Some((reg, ref mut use_idx)) = self.reg_map[chip_reg as usize] {
            // Already allocated. Happy path.
            *use_idx = self.use_index;

            return reg;
        }

        if !self.free_host_regs.is_empty() {
            // We've got a free host register left. Allocate it:
            let reg = self.free_host_regs.iter().next().unwrap();
            self.free_host_regs.remove(&reg);
            self.reg_map[chip_reg as usize] = Some((reg, self.use_index));
            debug!("alloc'd {:?} <-> V{:01X}", reg, chip_reg);

            if reg.is_callee_saved() {
                self.emit_reg_save(reg);
            }

            let offset = self.calc_offset(|state| &state.regs[chip_reg as usize]);
            self.emit_state_load_u8(reg, offset);

            return reg;
        }

        // No free host register. This means we have to spill an allocated register. We always spill
        // the least recently used register, which is the register with the lowest use index.
        let (spilled_chip8_reg, (spilled_host_reg, use_idx)) =
            self.reg_map.iter()
                        .enumerate()
                        .filter_map(|(i, item)| {
                            // only consider allocated CHIP-8 regs
                            if let Some(itm) = *item { Some((i, itm)) } else { None }
                        })
                        .min_by_key(|&(_, (_, used))| used) // get the least recently used one
                        .unwrap();                          // this must exist

        assert!(self.spill_chip8_reg(chip_reg));

        // ...and allocate the new one
        self.reg_map[chip_reg as usize] = Some((spilled_host_reg, self.use_index));

        let offset = self.calc_offset(|state| &state.regs[chip_reg as usize]);
        self.emit_state_load_u8(spilled_host_reg, offset);

        spilled_host_reg
    }

    /// Generates spill code writing the current value of the given CHIP-8 register into the
    /// `ChipState`. Then deallocates the host register assigned to it.
    ///
    /// Returns `true` if the register was spilled successfully and `false` if the register isn't
    /// allocated.
    fn spill_chip8_reg(&mut self, reg: u8) -> bool {
        // If the register isn't currently allocated to a host register, its value inside the
        // `ChipState` must already be correct.
        if let Some((host_reg, use_idx)) = self.reg_map[reg as usize].take() {
            debug!("spilling reg V{:01X}, allocated to {:?} with a use index of {}",
                reg, host_reg, use_idx);

            // Mark the reg as free
            self.free_host_regs.insert(host_reg);

            let offset = self.calc_offset(|state| &state.regs[reg as usize]);
            self.emit_state_store_u8(offset, host_reg);
            true
        } else {
            false
        }
    }

    /// Emits code for a function call, passing a number of integer arguments.
    ///
    /// Before a function call can be performed, we need write all allocated registers to the
    /// `ChipState` and deallocate them to ensure consistency.
    fn emit_call<C: Callable>(&mut self, callee: C, args: &[u64]) {
        assert_eq!(C::param_count() as usize, args.len());

        // Write registers to the `ChipState`, so the callee can access consistent data.
        for reg in 0..16 {
            self.spill_chip8_reg(reg);
        }

        // Save caller-saved registers in use
        // FIXME This differs between the 2 calling conventions, but if we save more than we need
        // to, it doesn't matter for correctness.
        self.emit_rsi_save();

        // Emit all arguments into the correct registers by emitting 64-bit immediate loads for
        // them.
        let regs = calls::get_arg_reg_codes();
        if regs.len() < args.len() {
            panic!("not enough registers for call arguments (got {} args, but {} regs)",
                args.len(), regs.len());
        }

        for (&arg, &reg) in args.iter().zip(regs) {
            self.emit_load_imm64(reg, arg);
        }

        // Emit the actual call. `call` doesn't support 64-bit immediates, so we load the 64-bit
        // address into `rax` and call that.
        self.emit_raw(&[0x48, 0xb8]);   // `movabs rax, <ADDRESS>`
        self.code_buffer.write_u64::<LittleEndian>(callee.get_addr());
        self.emit_raw(&[0xff, 0xd0]);   // `call rax`

        self.emit_rsi_restore();
    }

    fn state(&self) -> Ref<ChipState> {
        self.state_rc.borrow()
    }

    /// Get the address of the `ChipState` structure to use in the compiled code.
    fn state_address(&self) -> *const ChipState {
        &*self.state_rc.borrow()
    }

    /// Calculate an offset into the `ChipState`.
    fn calc_offset<T, F: FnOnce(&ChipState) -> &T>(&self, f: F) -> Offset<T> {
        let state = self.state();
        let addr = f(&state) as *const T as usize;

        let state_addr = self.state_address() as usize;
        assert!(state_addr <= addr);

        Offset {
            bytes: addr - state_addr,
            _p: PhantomData,
        }
    }

    /// Emits raw bytes as machine code. Use with care!
    ///
    /// Consult your local therapist for information about x86 instructon encoding.
    fn emit_raw(&mut self, code: &[u8]) {
        self.code_buffer.extend_from_slice(code);
    }

    /// Emits an immediate operation.
    ///
    /// See http://www.c-jump.com/CIS77/CPU/x86/X77_0210_encoding_add_immediate.htm
    fn emit_imm_op(&mut self, reg: X86Register, op: ImmediateOp, imm: u8) {
        self.emit_raw(&[0x80, (0b11 << 6) | ((op as u8) << 3) | reg.as_reg_field_value(), imm]);
    }

    /// Emits code to load a `u8` from the given `ChipState` field into a register.
    ///
    /// The generated code is safe if the address can be read safely during the time the code is
    /// executed. This is the case for anything stored in the `ChipState` and static variables.
    fn emit_state_load_u8(&mut self, reg: X86Register, field: Offset<u8>) {
        // We need to use `MOV r8, r/m8` for moving a byte from memory to a register, so the opcode
        // is `0x8A`. Since it uses an `r/m` operand, the next byte is a ModR/M byte, which is
        // explained at http://www.c-jump.com/CIS77/CPU/x86/X77_0060_mod_reg_r_m_byte.htm.
        //
        // For the `MOD` bits, we use `0b10`, which means that a 4-byte signed displacement follows
        // the addressing mode bytes. That byte will be set to whatever the offset into the
        // `ChipState` is. FIXME: If the offset is smaller than 128 Bytes we can just use the 1-byte
        // displacement, specified with `0b01`.
        //
        // The `REG` field contains the target register, so it depends on the passed `X86Register`.
        //
        // The `R/M` field specifies the 32-bit register containing the base address to load from,
        // so we need to set it to `esi` (`0b110`), since we store the address of the `ChipState` in
        // there.

        let offset = field.bytes;
        assert!(offset < i32::MAX as usize);

        self.emit_raw(&[0x8A, (0b10 << 6) | (reg.as_reg_field_value() << 3) | 0b110]);
        self.code_buffer.write_i32::<LittleEndian>(offset as i32).unwrap();
    }

    /// Emits code to store a `u8` from a register into a `ChipState` field.
    fn emit_state_store_u8(&mut self, field: Offset<u8>, reg: X86Register) {
        let offset = field.bytes;
        assert!(offset < i32::MAX as usize);

        self.emit_raw(&[0x88, (0b10 << 6) | (reg.as_reg_field_value() << 3) | 0b110]);
        self.code_buffer.write_i32::<LittleEndian>(offset as i32).unwrap();
    }

    /// Emits code to store a constant `u16` in a field inside the `ChipState`.
    fn emit_state_store_u16(&mut self, field: Offset<u16>, value: u16) {
        // This operation is equivalent to this assembly:
        //
        //     mov word ptr [esi + <field>], <value>
        //
        // Here, we're using `MOV r/m16, imm16`, so our opcode is `0xC7`. To set the operand size to
        // 16 bits, we need to use the Operand size prefix byte `0x66` (otherwise, this would be a
        // 32-bit operation).
        //
        // Then we encode, again, the ModR/M byte just like we did in `emit_state_load_u8`: The
        // `MOD` field is set just like before, but the `REG` field is left at `0`, since we only
        // have a single addressed operand (the other one is an immediate). `R/M` is set to the
        // encoding of `esi` - `0b110`.
        //
        // The field offset is the next thing to encode, as a signed 32-bit displacement. Then
        // follows the immediate we want to store.

        let offset = field.bytes;
        assert!(offset < i32::MAX as usize);

        self.emit_raw(&[0x66, 0xC7, (0b10 << 6) | (0b110)]);
        self.code_buffer.write_i32::<LittleEndian>(offset as i32).unwrap();
        self.code_buffer.write_u16::<LittleEndian>(value).unwrap();
    }

    /// Emits code for loading a constant `u8` value into a host register.
    fn emit_load_imm8(&mut self, reg: X86Register, value: u8) {
        // `mov`ing an 8-bit immediate into a register works with `0xB0 + reg` followed by the
        // immediate.

        self.emit_raw(&[0xB0 | reg.as_reg_field_value(), value]);
    }

    /// Emits code for loading a 64-bit immediate value into a 64-bit register, which is specified
    /// as the encoding to use in the opcode.
    ///
    /// I'd really like to generalize the register abstraction to work for this use case at some
    /// point.
    fn emit_load_imm64(&mut self, reg: u8, value: u64) {
        // Emit a `movabs`
        self.emit_raw(&[0x48, 0xB8 + reg]);
        self.code_buffer.write_u64::<LittleEndian>(value).unwrap();
    }

    /// Load a constant pointer value into `rsi`.
    fn emit_load_rsi_ptr<T>(&mut self, ptr: *mut T) {
        // FIXME This also needs 32-bit support

        // Emit a `movabs`
        self.emit_raw(&[0x48, 0xBE]);
        self.code_buffer.write_u64::<LittleEndian>(ptr as usize as u64).unwrap();
    }

    /// Emits code that will save the value of a host register.
    ///
    /// Note that this will do nothing if the value was already saved, so this can't be used to
    /// spill registers.
    fn emit_reg_save(&mut self, reg: X86Register) {
        // Since we can only push 32-bit registers, pushing the low 8 bits of a register will also
        // push the high 8 bits (and vice versa), so we add them both to the saved list.

        if self.saved_host_regs.contains(&reg) || self.saved_host_regs.contains(&reg.other_half()) {
            // Already saved as part of its other half, or this was called twice with the same reg,
            // which really shouldn't happen.
            debug!("reg {:?} already saved, skipping", reg);
            return;
        }

        // 0x50 = push r32
        // I think on x64 this will push the corresponding 64-bit register, but that should be fine,
        // too.
        self.emit_raw(&[0x50 | reg.as_r32_value()]);

        self.saved_host_regs.push(reg);
    }

    /// Emits code to restore all saved host registers.
    fn emit_restore_regs(&mut self) {
        for host_reg in self.saved_host_regs.iter().rev() {
            trace!("restore {:?}", host_reg);
            // `pop <REG>`
            self.code_buffer.extend_from_slice(&[0x58 | host_reg.as_r32_value()]);
        }
    }

    /// FIXME Botch, please remove
    /// FIXME FIXME We need to save `rdi` the same way!
    fn emit_rsi_save(&mut self) {
        self.emit_raw(&[0x56]); // `push rsi`
        self.emit_raw(&[0x57]); // `push rdi`
        // now not even the name makes sense... please refactor me :'(
    }

    fn emit_rsi_restore(&mut self) {
        self.emit_raw(&[0x5F]); // `pop rdi`
        self.emit_raw(&[0x5E]); // `pop rsi`
    }
}
