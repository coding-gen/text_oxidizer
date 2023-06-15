//! Byte Pair Encoder Tokenizer
//! Tools for Natural Language Processing in Rust
//!
//! Authors: Sawyer Norquist, Genevieve LaLonde
//! Programming in Rust 2023
//! Professor Dr. Bart Massey 2023
//! Resources:
//! https://doc.rust-lang.org/reference/
//! https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt
//! https://towardsdatascience.com/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0

use csv::{Reader, Writer};
use radsort::sort_by_key;
use std::{collections::HashMap, error::Error, ffi::OsStr};

pub use crate::debug_tools::*;
pub use crate::tokenize::*;

#[derive(Debug, Clone)]
struct WordCount {
    word: Vec<String>,
    count: u32,
}

#[derive(Debug, Clone)]
pub struct Frequency {
    token: String, // All the tokens form the vocab
    freq: u32,
}

// Init the vocab as all the lowercase letters and punctuation.
// Build the corpus as a list of letters for each word, and their counts.
fn init_vocab_corpus(token_lines: Vec<Vec<String>>) -> (Vec<Frequency>, Vec<WordCount>) {
    let mut word_freq: HashMap<String, u32> = HashMap::new();
    let mut corpus: HashMap<Vec<String>, u32> = HashMap::new();
    for line in token_lines {
        for token in line {
            let mut word: Vec<String> = Vec::new();
            for c in token.to_lowercase().chars() {
                word.push(c.to_string());

                // Build the frequency table with vocab of letters, and letter counts.
                *word_freq.entry(c.to_string()).or_insert(0) += 1;
            }
            word.push("</w>".to_string());

            // Create dict with counts of words using 'entry'
            // A dict is a more performant way to build this than vec of structs,
            // since we have to check for belonging on every word of the corpus.
            // Source https://stackoverflow.com/questions/64178272/what-is-the-idiomatic-rust-way-to-build-a-hashmap-of-character-counts
            *corpus.entry(word).or_insert(0) += 1;
        }
    }
    // Convert both tables after creation, since we don't need fast lookup after this.
    // source:
    // https://stackoverflow.com/questions/71369758/cast-hashmap-to-vector
    let frequency_table: Vec<Frequency> = word_freq
        .into_iter()
        .map(|x| Frequency {
            token: x.0,
            freq: x.1,
        })
        .collect();

    let corp2: Vec<WordCount> = corpus
        .into_iter()
        .map(|x| WordCount {
            word: x.0,
            count: x.1,
        })
        .collect();
    (frequency_table, corp2)
}

// Locate the bigram with highest occurence in the corpus
fn get_max_freq_bigram(corpus: &Vec<WordCount>) -> (Vec<String>, u32) {
    /*
    TODO: improve performance.
    This can be optimized, by storing counts in a table
    Update the table for the constituent parts, when the bigram pair to merge is selected
    returna candy, and pass it back on next call

    extract out the construction of candidates, only do it once.
    after building candidates, sort it in descending order of count
    Then on each loop iteration, just pop the first element.
    */
    let mut candidate_bigram: (Vec<String>, u32) = (vec!["".to_string(), "".to_string()], 0);
    let mut candidates: HashMap<Vec<String>, u32> = HashMap::new();

    for grams_list in corpus {
        for i in 0..grams_list.word.len() - 1 {
            // Count up the candidate bigrams
            *candidates
                .entry(vec![
                    grams_list.word[i].clone(),
                    grams_list.word[i + 1].clone(),
                ])
                .or_insert(0) += grams_list.count;

            // Find the max occurence bigram.
            if candidates[&vec![grams_list.word[i].clone(), grams_list.word[i + 1].clone()]]
                > candidate_bigram.1
            {
                let k = vec![grams_list.word[i].clone(), grams_list.word[i + 1].clone()];
                let count = candidates[&k];
                candidate_bigram = (k, count);
            }
        }
    }
    // Convert to vector string for sorting, and reuse.
    let /*mut*/ _candy: Vec<WordCount> = candidates.into_iter().map(|x| WordCount { word: x.0, count: x.1 }).collect();
    candidate_bigram
}

/// At each bigram selected to merge,
/// reduce the count of constituent grams,
/// and add the new token.
fn update_frequency_table(
    mut frequency_table: Vec<Frequency>,
    max_bigram: &[String],
    bigram_count: &u32,
) -> Vec<Frequency> {
    let mut i = 0;
    while i < frequency_table.len() {
        let mut reduced = false;
        if frequency_table[i].token == max_bigram[0] || frequency_table[i].token == max_bigram[1] {
            frequency_table[i].freq -= bigram_count;
            if frequency_table[i].freq == 0 {
                frequency_table.remove(i);
                reduced = true;
            }
        }
        // Vec size was adjusted, so adjust the index.
        if !reduced {
            i += 1;
        }
    }
    // Add the new token to the table.
    frequency_table.push(Frequency {
        token: [max_bigram[0].clone(), max_bigram[1].clone()].join(""),
        freq: *bigram_count,
    });
    frequency_table
}

// Merge all instances of the bigram in the corpus.
fn merge_bigrams(mut corpus: Vec<WordCount>, max_bigram: String) -> Vec<WordCount> {
    for entry in &mut corpus {
        //let grams_list: Vec<String> = grams_and_count.0.to_vec();
        let mut j: usize = 0;
        while j < entry.word.len() - 1 {
            let two_gram = [entry.word[j].clone(), entry.word[j + 1].clone()].join("");

            if two_gram == max_bigram {
                // merge
                let mut merged_grams_list: Vec<String> = Vec::new();
                merged_grams_list.extend_from_slice(&entry.word[..j]);
                merged_grams_list.push(two_gram);
                merged_grams_list.extend_from_slice(&entry.word[j + 2..]);
                entry.word = merged_grams_list;
                j += 1;
            }
            j += 1;
        }
    }
    corpus
}

// Use byte-pair encoding to build a vocabulary of size n,
// according to statistical frequency of bi-gram.
pub fn bpe_training(token_lines: Vec<Vec<String>>, mut n: u8) -> Vec<String> {
    // A common value of n: 50,000
    let min_corpus_size = 52;
    if n < min_corpus_size {
        n = min_corpus_size;
    }
    let (mut frequency_table, mut corpus) = init_vocab_corpus(token_lines);

    // Loop over corpus, expanding vocab with next most likely bigram,
    // until desired vocab size reached.
    while frequency_table.len() < n.into() {
        // Locate bigram to merge.
        let (max_bigram, bigram_count) = get_max_freq_bigram(&corpus);

        // Exit early if no more mergeable bigrams.
        if [max_bigram[0].clone(), max_bigram[1].clone()].join("") == "" {
            break;
        }

        // Update the frequency table
        frequency_table = update_frequency_table(frequency_table, &max_bigram, &bigram_count);

        // Update the corpus
        corpus = merge_bigrams(
            corpus,
            [max_bigram[0].clone(), max_bigram[1].clone()].join(""),
        );
    }
    let mut vocab: Vec<String> = Vec::new();
    for entry in &frequency_table {
        vocab.push(entry.token.clone());
    }
    vocab
}

pub fn bpe_encoding(
    text_lines: Vec<Vec<String>>,
    mut vocab_lines: Vec<Vec<String>>,
) -> Vec<Vec<String>> {
    //iterate over vocab
    // order longest token of vocab to shortest
    let mut formatted_vocab: Vec<String> = Vec::new();
    // Sort by token length, not including the end of word indicator.
    // source: https://docs.rs/radsort/latest/radsort/fn.sort_by_key.html
    sort_by_key(&mut vocab_lines, |s| s[0].len());
    for entry in &mut vocab_lines {
        if entry.len() == 2 {
            formatted_vocab.push(format!("{}{}", &entry[0], "</w>"));
        } else {
            formatted_vocab.push(entry[0].clone());
        }
    }

    let mut formatted_seqs: Vec<Vec<String>> = Vec::new();
    // process input text and add </w> to the end of each word
    for sequence in &text_lines {
        let mut tmp_line: Vec<String> = Vec::new();
        for token in sequence {
            tmp_line.push(format!("{}{}", token, "</w>"));
        }
        formatted_seqs.push(tmp_line);
    }
    /*
    Not yet implemented:
    for each token of the input sequence,
    at every position of the token,
    compare to every token of the vocab.
    Walk the vocab in reverse order.

    If a match is found, replace.

    Any substrings left in the input are replaced by </unknown> for now

    Future:
    retrain bpe with these new words,
    add results to the vocab,
    and tokenize the unknown words.
    */

    formatted_seqs
}

/// Accepts a path to a CSV file.
/// Returns a Vec of Vec of basic tokenized Strings for further processing by BPE, or any resultant errors.
pub fn parse_csv_to_tokens(fpath: &OsStr) -> Result<Vec<Vec<String>>, Box<dyn Error>> {
    let mut out: Vec<Vec<String>> = Vec::new();
    let mut reader = Reader::from_path(fpath)?;

    for result in reader.byte_records() {
        let record = result?;
        let tokens =
            tokenize_line_alphas_lowercase(&String::from_utf8_lossy(record.get(1).unwrap()));
        out.push(tokens)
    }
    Ok(out)
}

/// Accepts a path to a CSV file.
/// Returns a Vec of Sequences to be encoded by BPE, or any resultant errors.
pub fn parse_csv_to_lines(fpath: &OsStr) -> Result<Vec<String>, Box<dyn Error>> {
    let mut out: Vec<String> = Vec::new();
    let mut reader = Reader::from_path(fpath)?;

    for result in reader.byte_records() {
        let record = result?;
        let sequence = String::from_utf8_lossy(record.get(1).unwrap()).to_string();
        out.push(sequence)
    }
    Ok(out)
}

/// Accepts a path to a CSV file.
/// Returns a Vec of Vec of Strings for further processing by BPE, or any resultant errors.
pub fn parse_txt_to_tokens(fpath: &OsStr) -> Result<Vec<Vec<String>>, Box<dyn Error>> {
    let mut out: Vec<Vec<String>> = Vec::new();
    let mut reader = Reader::from_path(fpath)?;

    for result in reader.byte_records() {
        let record = result?;
        let tokens =
            tokenize_line_alphas_lowercase(&String::from_utf8_lossy(record.get(0).unwrap()));
        out.push(tokens)
    }
    Ok(out)
}

/// Accepts a path to a CSV file.
/// Returns a Vec of Vec of Strings for further processing by BPE, or any resultant errors.
pub fn parse_txt_to_lines(fpath: &OsStr) -> Result<Vec<String>, Box<dyn Error>> {
    let mut out: Vec<String> = Vec::new();
    let mut reader = Reader::from_path(fpath)?;

    for result in reader.byte_records() {
        let record = result?;
        let sequence = String::from_utf8_lossy(record.get(0).unwrap()).to_string();
        out.push(sequence)
    }
    Ok(out)
}

/// Takes a filepath as an &OsStr and a &Vec<String> to save into a TXT
/// Returns an error if one occurs
pub fn save_bpe_vocab(fpath: &OsStr, to_save: &Vec<String>) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_path(fpath)?;
    wtr.write_record(["Tokens in vocab:"])?;

    for token in to_save {
        wtr.write_record([token])?;
    }
    Ok(())
}

/// Takes a filepath as an &OsStr and a &Vec<String> to save into a TXT
/// Returns an error if one occurs
pub fn save_bpe_encoding(fpath: &OsStr, to_save: &Vec<Vec<String>>) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_path(fpath)?;
    wtr.write_record(["Tokenized sequences"])?;

    for line in to_save {
        for token in line {
            wtr.write_record([token])?;
        }
    }
    Ok(())
}
