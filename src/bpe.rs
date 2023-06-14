use std::{collections::HashMap, error::Error, ffi::OsStr};
use csv::{Reader, Writer};

pub use crate::tokenize::*;
pub use crate::debug_tools::*;


#[derive(Debug, Clone)]
struct WordCount {
    word: Vec<String>,
    count: u32,
}

#[derive(Debug, Clone)]
pub struct Frequency {
    token: String, // equivalent to the entries of the vocab
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
            // TODO this is supposed to break up prefix/suffix. Why isn't it?
            // Because we are expected to know when to stop training before those tokens get merged.
            word.push("</w>".to_string());

            // Create dict with counts of words using 'entry'
            // A dict is a more performant way to do this than vec of structs, 
            // since we have to check for belonging on every word of the corpus.
            // Source https://stackoverflow.com/questions/64178272/what-is-the-idiomatic-rust-way-to-build-a-hashmap-of-character-counts

            // TODO add end gram </w> to the end of words.
            // to differentiate suffixes from pre/in-fixes
            *corpus.entry(word).or_insert(0) += 1;
        }
    }
    // Convert both tables after creation, since we don't need fast lookup after this.
    // source:
    // https://stackoverflow.com/questions/71369758/cast-hashmap-to-vector
    let frequency_table: Vec<Frequency> = 
        word_freq.into_iter().map(|x| Frequency { token: x.0, freq: x.1 }).collect();

    let corp2: Vec<WordCount> = 
        corpus.into_iter().map(|x| WordCount { word: x.0, count: x.1 }).collect();
    println!("vocab:");
    for w in &frequency_table{
        println!("{:?}", w);
    }
    println!("corpus:");
    for w in &corp2{
        println!("{:?}", w);
    }
    (frequency_table, corp2)
}


// Locate the n-gram with highest occurence in the corpus
fn get_max_freq_ngram(corpus: &Vec<WordCount>) -> (Vec<String>, u32) {
    /*
    This can be optimized, by storing counts in a table
    Update the table for the constituent parts, when the n-gram pair to merge is selected
    */
    let mut candidate_n_gram: (Vec<String>, u32) = (vec!["".to_string(), "".to_string()], 0);
    let mut candidates: HashMap<Vec<String>, u32> = HashMap::new();

    // corpus entries: grams_list is a WordCount
    //(grams_list.word, grams_list.count) in &corpus

    // TODO extract out the construction of candidates, only do it once.
    for grams_list in corpus {
        
        for i in 0..grams_list.word.len()-1 {
            //let two_gram = vec![grams_list.word[i], grams_list.word[i+1]];
            //let two_gram = [grams_list.word[i].clone(), grams_list.word[i+1].clone()].join("");

            // TODO don't build candidates every time. 
            // Pass it around somehow, and just update it. 

            // Count up the candidate n-grams

            *candidates.entry(vec![grams_list.word[i].clone(), grams_list.word[i+1].clone()]).or_insert(0) += grams_list.count;

            if candidates[&vec![grams_list.word[i].clone(), grams_list.word[i+1].clone()]] > candidate_n_gram.1 {
                let k = vec![grams_list.word[i].clone(), grams_list.word[i+1].clone()];
                let count = candidates[&k];
                candidate_n_gram = (k, count);
            }

            /*
            // Find the max candidate on the fly while building candidates.
            if candidates[&[grams_list.word[i].clone(), grams_list.word[i+1].clone()].join("")] > candidate_n_gram.1 {
                let k = [grams_list.word[i].clone(), grams_list.word[i+1].clone()].join("");
                let count = candidates[&k];
                candidate_n_gram = (k, count);
            }*/
        }
    }
    let /*mut*/ _candy: Vec<WordCount> = candidates.into_iter().map(|x| WordCount { word: x.0, count: x.1 }).collect();
    // TODO pass candy back, and use as param on the way back in.
    // after building candidates, sort it in descending order of count
    // Then on each loop iteration, just pop the first element.

    candidate_n_gram
}


// Merge all instances of the n-gram in the corpus.
fn merge_ngrams(mut corpus: Vec<WordCount>, max_n_gram: String) -> Vec<WordCount> {
    for i in 0..corpus.len() {
        //let grams_list: Vec<String> = grams_and_count.0.to_vec();
        let mut j: usize = 0;
        while j < corpus[i].word.len()-1 {

            let two_gram = [corpus[i].word[j].clone(), corpus[i].word[j+1].clone()].join("");

            if two_gram == max_n_gram {
                // merge
                let mut merged_grams_list: Vec<String> = Vec::new();
                merged_grams_list.extend_from_slice(&corpus[i].word[..j]);
                merged_grams_list.push(two_gram);
                merged_grams_list.extend_from_slice(&corpus[i].word[j+2..]);
                corpus[i].word = merged_grams_list;
                j += 1;
            } 
            j += 1;
        }
    } 
    corpus
}


// Use byte-pair encoding to build a vocabulary of size n,
// according to statistical frequency of n-grams.
pub fn bpe_training(token_lines: Vec<Vec<String>>, mut n: u8) -> Vec<String> {
    // A common value of n: 50,000
    let min_corpus_size = 52;
    if n < min_corpus_size {
        n = min_corpus_size;
    }
    let (mut frequency_table, mut corpus) = init_vocab_corpus(token_lines);

    // Loop over corpus, expanding vocab with next most likely n-gram,
    // until desired vocab size reached.
    while frequency_table.len() < n.into() {
        let (max_n_gram, n_gram_count) = get_max_freq_ngram(&corpus);

        // Exit early if no more mergeable n-grams.
        if [max_n_gram[0].clone(), max_n_gram[1].clone()].join("") == "" {
            break;
        }

        // Update the frequency table
        // TODO extract this to its own function.
        let mut i = 0;

        while i < frequency_table.len() {
            let mut reduced = false;
            if frequency_table[i].token == max_n_gram[0] || frequency_table[i].token == max_n_gram[1] {
                //let prev_freq = frequency_table[i].freq;
                frequency_table[i].freq -=  n_gram_count;
                if frequency_table[i].freq == 0 {
                    frequency_table.remove(i);
                    reduced = true;
                }
            }
            if !reduced {
                i += 1;
            }
        }
        // Add the new token to the table.
        frequency_table.push(Frequency {token: [max_n_gram[0].clone(), max_n_gram[1].clone()].join(""), freq: n_gram_count});

        // Update the corpus
        corpus = merge_ngrams(corpus, [max_n_gram[0].clone(), max_n_gram[1].clone()].join(""));
    }
    let mut vocab: Vec<String> = Vec::new();
    for entry in &frequency_table{
        vocab.push(entry.token.clone());
    }
    vocab
}

/*
fn bpe_encoding(token_lines: Vec<Vec<String>>) {
    //iterate over vocab
    // order longest token of vocab to shortest
    // Any substrings left in the input are replaced by </unknown> for now
    // retrain bpe with these new words, 
    // add results to the vocab, 
    // and tokenize the unknown words.
}
*/

/// Accepts a path to a CSV file.
/// Returns a Vec of Vec of Strings for further processing by BPE, or any resultant errors.
pub fn parse_csv_to_lines(fpath: &OsStr) -> Result<Vec<Vec<String>>, Box<dyn Error>> {
    let mut out: Vec<Vec<String>> = Vec::new();
    let mut reader = Reader::from_path(fpath)?;

    for result in reader.byte_records() {
        let record = result?;
        let tokens =
            tokenize_line_alphas_lowercase(&String::from_utf8_lossy(record.get(1).unwrap()));
        //let target = String::from_utf8_lossy(record.get(0).unwrap()).into_owned();

        //let newout = LineTarget { tokens, target };

        out.push(tokens)
    }

    Ok(out)
}

/// Takes a filepath as an &OsStr and a &Vec<String> to save into a CSV
/// Returns an error if one occurs
pub fn save_bpe_vocab(
    fpath: &OsStr,
    to_save: &Vec<String>,
) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_path(fpath)?;
    wtr.write_record(["Tokens in vocab:"])?;

    for token in to_save {
        wtr.write_record([
            token
        ])?;
    }
    Ok(())
}