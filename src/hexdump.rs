pub fn hexdump(s: &[u8]) -> String {
    format!("[ {} ]", s.iter().map(|b| format!("{:02X}", b)).collect::<Vec<_>>().join(", "))
}
