const START: i64 = 5431;
const MULT:  i64 = 847209285431;
const SHIFT: i64 = 5;

pub fn hash(text: &str) -> i64 {
    let mut hash: i64 = START;
    for byte in text.bytes() {
        hash = (hash << SHIFT)
            .wrapping_add(hash)
            .wrapping_mul(MULT)
            .wrapping_add(byte as i64);
    }
    hash = hash
        .wrapping_add(hash)
        .wrapping_mul(MULT);
    hash
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_use_hash() {
        let hello = hash("hello");
        dbg!(hello);
        let hellp = hash("hellp");
        dbg!(hellp);
        let herro = hash("herro");
        dbg!(herro);
        assert!(hello != hellp);
        assert!(hello != herro);
        assert!(herro != hellp);
        assert!(hello < hellp);
    }
    #[test]
    fn test_smoke() {
        let test_use_hash = "let out smoke test";
        let out = hash(test_use_hash);
        dbg!(out);
    }
    #[test]
    fn test_u8_wrapping() {
        let number: u8 = 255;
        let number: u8 = number.wrapping_add(1);
        assert!(number == 0);
    }
}