#[macro_use] extern crate log;
extern crate cpal;
extern crate env_logger;
extern crate byteorder;
extern crate memmap;
extern crate clap;
extern crate fnv;
extern crate enum_set;
extern crate minifb;

mod cache;
mod calls;
mod chip8;
mod ext;
mod hexdump;
mod jit;
mod reg;

use chip8::Chip8;
use clap::{App, Arg, ArgMatches};

use std::fs::File;
use std::io::{self, Read};

fn run(args: &ArgMatches) -> io::Result<()> {
    let rom_path = args.value_of("rom").unwrap();
    let mut file = try!(File::open(rom_path));
    let mut rom = Vec::new();
    try!(file.read_to_end(&mut rom));

    Chip8::new(&rom).run();

    Ok(())
}

fn main() {
    env_logger::init().unwrap();

    let args = App::new("chip-8")
        .arg(Arg::with_name("rom")
            .takes_value(true)
            .required(true))
        .get_matches();

    match run(&args) {
        Ok(()) => {}
        Err(e) => {
            println!("error: {}", e);
        }
    }
}
