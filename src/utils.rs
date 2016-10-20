use std::fmt::UpperHex;

macro_rules! errprintln {
    ( $($args:tt)* ) => {{
        use std::io::Write;
        let mut stderr = ::std::io::stderr();
        writeln!(stderr, $($args)*).unwrap();
    }};
}

pub fn hexdump<T: UpperHex>(s: &[T]) -> String {
    format!("[ {} ]", s.iter().map(|b| format!("{:02X}", b)).collect::<Vec<_>>().join(", "))
}
