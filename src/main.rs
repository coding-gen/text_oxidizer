use csv::Reader;
use lazy_static::lazy_static;
use prompted::input;
use regex::Regex;
use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};

mod debug_tools;
pub use crate::debug_tools::*;

/// Print a passed usage error message and exit.
/// Will panic instead if in test configuration.
fn error(err: &str) -> ! {
    eprintln!("{}", err);
    #[cfg(not(test))]
    std::process::exit(1);
    #[cfg(test)]
    panic!("error");
}

//Struct for pairing a token stream to a target value for training models
#[derive(Debug)]
struct LineTarget {
    tokens: Vec<String>,
    target: String,
}

//Struct for pairing a number of target matches and token occurances for Bayes analysis
#[derive(Debug)]
struct TokenOccurence {
    matches: usize,
    occurences: usize,
}

fn create_bayes(input: Vec<LineTarget>, target: &str) -> HashMap<String, TokenOccurence> {
    let mut bayes: HashMap<String, TokenOccurence> = HashMap::new();
    for line in input {
        for token in line.tokens {
            if !bayes.contains_key(&token) {
                bayes.insert(
                    token.clone(),
                    TokenOccurence {
                        matches: 0,
                        occurences: 0,
                    },
                );
            }
            let inplace = bayes.get_mut(&token).unwrap();
            inplace.occurences += 1;
            if line.target == target {
                inplace.matches += 1;
            }
        }
    }

    bayes
}

//Accepts a path to a CSV file.  Just hands up any errors it recieves.
//Otherwise returns a Vec of LineTargets for further processing.
fn parse_csv_to_linetarget(fpath: &OsStr) -> Result<Vec<LineTarget>, Box<dyn Error>> {
    let mut out = Vec::new();
    let mut reader = Reader::from_path(fpath)?;

    for result in reader.byte_records() {
        let record = result?;
        let tokens =
            tokenize_line_alphas_lowercase(&String::from_utf8_lossy(record.get(1).unwrap()));
        let target = String::from_utf8_lossy(record.get(0).unwrap()).into_owned();

        let newout = LineTarget { tokens, target };

        out.push(newout)
    }

    Ok(out)
}

//https://users.rust-lang.org/t/how-to-return-bufreader/34651/6
///Accepts a file path and returns a Result containing either a BufReader or an IO error
fn open_reader(fpath: &OsStr) -> Result<BufReader<Box<dyn Read>>, Box<dyn Error>> {
    let fileobj = File::open(fpath)?;
    Ok(BufReader::new(Box::new(fileobj)))
}

//https://docs.rs/regex/latest/regex/
//Using lazy_static as recommended by regex crate docs
//Ignores invalid returns from the regex
fn tokenize_line(line: &str) -> Vec<String> {
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

//https://docs.rs/regex/latest/regex/
//Using lazy_static as recommended by regex crate docs
//Ignores invalid returns from the regex
//Only returns alpha sequences, including apostraphies
fn tokenize_line_alphas_lowercase(line: &str) -> Vec<String> {
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

// Takes in a Buffered Reader and returns a Vec<String> of the tokens found
//Errors if the BufReader contains invalid information.
fn tokenize_reader(filein: BufReader<Box<dyn Read>>) -> Vec<String> {
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

fn main() {
    // let test_text = "Here is some test text. Question though, \n Does it do what we want?";

    // for line in test_text.lines() {
    //     println!("Result from tokenizer: {:?}", tokenize_line(&line));
    // }
    // test_open_reader();
    // test_tokenize_line();
    // test_tokenize_reader();
    // test_parse_csv_to_linetarget();
    test_create_bayes();
}

//Test functions.  May or may not be unit tests.

//Assumes you have a test.txt in the current crate location
//Reads test.txt line by line, printing each line
fn test_open_reader() {
    let mut filepath = env::current_dir().unwrap();
    filepath.push("test.txt");
    let ostringpath = filepath.into_os_string();
    if let Ok(_seepath) = ostringpath.clone().into_string() {
        // println!("{}", seepath);
    } else {
        println!("Filepath not displayable - continuing...")
    }
    let nreader = open_reader(&ostringpath);

    if let Ok(linereader) = nreader {
        for _line in linereader.lines() {
            // println!("{}", line.unwrap())
        }
    } else {
        println!("Invalid filepath")
    }
}

//Tokenizes test.txt by looping through each line with tokenize_line.
//Assumes you have a test.txt in the current crate location
fn test_tokenize_line() {
    let mut filepath = env::current_dir().unwrap();
    let mut outvec: Vec<String> = Vec::new();
    filepath.push("test.txt");
    let ostringpath = filepath.into_os_string();
    if let Ok(_seepath) = ostringpath.clone().into_string() {
        // println!("{}", seepath);
    } else {
        println!("Filepath not displayable - continuing...")
    }
    let nreader = open_reader(&ostringpath);

    if let Ok(linereader) = nreader {
        for line in linereader.lines() {
            outvec.append(&mut tokenize_line(&line.unwrap()))
        }
    } else {
        println!("Invalid filepath")
    }

    for s in outvec.iter() {
        println!("{}", s);
    }
}

//Basically same as test_tokenize_line, except hands the reader to tokenize_reader
//Assumes you have a test.txt in the current crate location
fn test_tokenize_reader() {
    let mut filepath = env::current_dir().unwrap();
    let mut outvec: Vec<String> = Vec::new();
    filepath.push("test.txt");
    let ostringpath = filepath.into_os_string();
    if let Ok(_seepath) = ostringpath.clone().into_string() {
        //println!("{}", seepath);
    } else {
        println!("Filepath not displayable - continuing...")
    }
    let nreader = open_reader(&ostringpath);

    if let Ok(linereader) = nreader {
        outvec.append(&mut tokenize_reader(linereader))
    } else {
        println!("Invalid filepath")
    }

    for s in outvec.iter() {
        println!("{}", s);
    }
}

fn test_parse_csv_to_linetarget() {
    let mut filepath = env::current_dir().unwrap();
    filepath.push("Twitter-sentiment-self-drive-DFE-Test.csv");

    let ostringpath = filepath.into_os_string();
    if let Ok(_seepath) = ostringpath.clone().into_string() {
        //println!("{}", seepath);
    } else {
        println!("Filepath not displayable - continuing...")
    }

    let outvec = parse_csv_to_linetarget(&ostringpath).unwrap();

    for i in 0..8 {
        println!("{:?}", outvec.get(i))
    }
}

fn test_create_bayes() {
    let mut filepath = env::current_dir().unwrap();
    let target = "not_relevant";
    filepath.push("Twitter-sentiment-self-drive-DFE-Test.csv");

    let ostringpath = filepath.into_os_string();
    if let Ok(_seepath) = ostringpath.clone().into_string() {
        //println!("{}", seepath);
    } else {
        println!("Filepath not displayable - continuing...")
    }

    let outvec = parse_csv_to_linetarget(&ostringpath).unwrap();
    let bayes = create_bayes(outvec, target);

    for line in bayes.iter() {
        println!("Key: {}", line.0);
        println!("TokenOccurence: {:?}", line.1);
        if input!("End?") == "Q" {
            std::process::exit(0);
        }
    }
}
