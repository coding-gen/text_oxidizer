#[allow(unused)]
use std::{
    env,
    error::Error,
    ffi::OsStr,
    fs::File,
    io::{BufRead, BufReader, Read},
};

use lazy_static::lazy_static;
use regex::Regex;

/// Print a passed usage error message and exit.
/// Will panic instead if in test configuration.
fn error(err: &str) -> ! {
    eprintln!("{}", err);
    #[cfg(not(test))]
    std::process::exit(1);
    #[cfg(test)]
    panic!("error");
}

//  Based on https://users.rust-lang.org/t/how-to-return-bufreader/34651/6
/// Accepts a file path and returns a Result containing either a BufReader or an IO error
pub fn open_reader(fpath: &OsStr) -> Result<BufReader<Box<dyn Read>>, Box<dyn Error>> {
    let fileobj = File::open(fpath)?;
    Ok(BufReader::new(Box::new(fileobj)))
}

//  Based onhttps://docs.rs/regex/latest/regex/
//  Using lazy_static as recommended by regex crate docs
/// Accepts an &str to be broken down into word and punctuation token.
/// Output is a Vec<String> representing the token stream.
/// Returns apostrophy's inside of a word group, if one exists.  Case is maintained.
/// Ignores invalid returns from the regex
pub fn tokenize_line(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    lazy_static! {
        static ref REGTOKEN: Regex =
            Regex::new(r#"[[:alpha:]']+|[0-9]+|[?,.!:"=_\-%#@\&\]\)]"#).unwrap();
    }
    for cap in REGTOKEN.captures_iter(line) {
        let text = &cap[0];

        tokens.push(text.to_owned());
    }
    tokens
}

//  https://docs.rs/regex/latest/regex/
//  Using lazy_static as recommended by regex crate docs
/// Accepts an &str to be broken down into word tokens
/// Output is a Vec<String> representing the token stream.
/// Ignores invalid returns from the regex
/// Only returns alpha sequences, including apostrophies.
/// Resulting tokens are all lower case.
pub fn tokenize_line_alphas_lowercase(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    lazy_static! {
        static ref REGTOKEN: Regex = Regex::new(r#"[[:alpha:]']+"#).unwrap();
    }
    for cap in REGTOKEN.captures_iter(line) {
        let text = &cap[0];

        tokens.push(text.to_lowercase().to_owned());
    }
    tokens
}

/// Takes in a Buffered Reader and returns a Vec<String> of the tokens found using tokenize_line().
/// Panics if the BufReader contains invalid information.
pub fn tokenize_reader(filein: BufReader<Box<dyn Read>>) -> Vec<String> {
    let mut outvec: Vec<String> = Vec::new();
    for line in filein.lines() {
        if let Ok(p) = line {
            outvec.append(&mut tokenize_line(&p));
        } else {
            error("tokenize_reader: Bad output from BufReader");
        }
    }
    outvec
}

/// Ensures test.csv opens and is read correctly by comparing to pre-determined input.
/// Expects test.csv with proper contents to be in the root directory of the crate.
/// Future update to create a temporary file with the correct contents and use this to test.
#[test]
fn test_open_reader() {
    let mut filepath = env::current_dir().unwrap();
    filepath.push("test.csv");
    let ostringpath = filepath.into_os_string();
    let nreader = open_reader(&ostringpath).unwrap();
    let mut scan = Vec::new();

    for line in nreader.lines() {
        scan.push(line.unwrap());
    }

    assert_eq!(scan[1], r#"a,"Test, this is.""#);
    assert_eq!(scan[2], r#"b,second line"#);
}

/// Ensures test.csv tokenizes properly by comparing output of tokenize_line() to pretedermined output.
/// Expects test.csv with proper contents to be in the root directory of the crate.
/// Future update to create a temporary file with the correct contents and use this to test.
#[test]
fn test_tokenize_line() {
    let outvec = tokenize_line("Test line, should be bee's knees!");
    let compvec = ["Test", "line", ",", "should", "be", "bee's", "knees", "!"];

    assert_eq!(outvec.len(), compvec.len());

    for i in 0..outvec.len() {
        assert_eq!(outvec[i], compvec[i]);
    }
}

/// Ensures test.csv tokenizes properly via reader by comparing output of tokenize_reader() to pretedermined output.
/// Expects test.csv with proper contents to be in the root directory of the crate.
/// Future update to create a temporary file with the correct contents and use this to test.
#[test]
fn test_tokenize_reader() {
    let mut filepath = env::current_dir().unwrap();
    let mut outvec: Vec<String> = Vec::new();
    filepath.push("test.csv");
    let ostringpath = filepath.into_os_string();
    let nreader = open_reader(&ostringpath).unwrap();

    let line = [
        "target", ",", "line", "a", ",", r#"""#, "Test", ",", "this", "is", ".", r#"""#, "b", ",",
        "second", "line",
    ];

    outvec.append(&mut tokenize_reader(nreader));

    assert_eq!(line.len(), outvec.len());

    for i in 0..line.len() {
        assert_eq!(line[i], outvec[i]);
    }
}
