use stop_words;
use rust_stemmers::{Algorithm, Stemmer};
use std::collections::HashSet;

pub struct Cleaner {
    stemmer: Stemmer,
    stop_words: HashSet<String>,
}

impl Cleaner {
    pub fn new() -> Self {
        let stemmer = Stemmer::create(Algorithm::English);
        let stop_words = stop_words::get(stop_words::LANGUAGE::English);
        let stop_words: HashSet<String> = stop_words
            .iter()
            .map(|s| s.to_string())
            .collect();
        Self {
            stemmer,
            stop_words,
        }
    }

    pub fn clean(&self, text: &str) -> String {
        let words: Vec<String> = text
            .split_whitespace()
            .map(|word| {
                word.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_lowercase()
            })
            .filter(|word| !word.is_empty())
            // Remove "filler words" "this, the, and, a, uh"
            .filter(|word| !self.stop_words.contains(word))
            // Remove suffix "ing, less, ness, er"
            .map(|word| self.stemmer.stem(&word).to_string())
            .collect();
        words.join(" ")
    }
}