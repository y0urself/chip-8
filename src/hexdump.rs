use std::fmt::UpperHex;

pub fn hexdump<T: UpperHex>(s: &[T]) -> String {
    format!("[ {} ]", s.iter().map(|b| format!("{:02X}", b)).collect::<Vec<_>>().join(", "))
}
