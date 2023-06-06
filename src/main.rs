// use core::num::dec2flt::number::Number;
use csv::{Reader, Writer};
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

fn user_exit() {
    if input!("Continue? Enter 'N' to exit: ").to_uppercase() == "N" {
        std::process::exit(0);
    }
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
    class_a: i32,
    class_b: i32,
}

//Pair token to class occurence probabilities
//For use in Naive Bayes
#[derive(Debug)]
struct TokenProbabilities {
    class_a: f64,
    class_b: f64,
}

//Captures number of words in each class
//For use in Naive Bayes
#[derive(Debug)]
struct NumberWords {
    class_a: i32,
    class_b: i32,
}

// Takes in a vec of LineTarget and the target to match against.
// Outputs a HashMap of TokenOccurence for further processing.
// Intended for use in Naive Bayes.
fn bayes_preprocess(
    input: &Vec<LineTarget>,
    target: &str,
) -> (HashMap<String, TokenOccurence>, NumberWords) {
    let mut occurence: HashMap<String, TokenOccurence> = HashMap::new();
    let mut numwords = NumberWords {
        class_a: 0,
        class_b: 0,
    };
    for line in input {
        for token in &line.tokens {
            if !occurence.contains_key(token) {
                occurence.insert(
                    token.clone(),
                    TokenOccurence {
                        class_a: 0,
                        class_b: 0,
                    },
                );
            }
            let inplace = occurence.get_mut(token).unwrap();
            if line.target == target {
                inplace.class_a += 1;
                numwords.class_a += 1;
            } else {
                inplace.class_b += 1;
                numwords.class_b += 1;
            }
        }
    }

    (occurence, numwords)
}

//  Takes in output of the preprocessor - a hashmap of string and occurance pairs, and the class wordcounts.
//  Outputs a hashmap of strings and associated class probabilities.
//  Should look at float math, and the associated i32 variables.  Some datasets may have extremely large wordcount.
//  Could potentially handle by chunking out the processing and aggregating the resulting probabilities.
fn generate_naive_bayes_model(
    input: &HashMap<String, TokenOccurence>,
    wordcount: NumberWords,
) -> HashMap<String, TokenProbabilities> {
    let mut bayes = HashMap::new();
    for item in input {
        let probabilities = TokenProbabilities {
            class_a: f64::from(item.1.class_a) / f64::from(wordcount.class_a),
            class_b: f64::from(item.1.class_b) / f64::from(wordcount.class_b),
        };
        bayes.insert(item.0.clone(), probabilities);
    }

    bayes
}

fn generate_naive_bayes_class_probability(
    model: &HashMap<String, TokenProbabilities>,
    line: &Vec<String>,
) -> TokenProbabilities {
    let mut class_a = 1_f64;
    let mut class_b = 1_f64;

    for token in line {
        if let Some(result) = model.get(token) {
            // println!("{}", token);
            // println!("{}, {}", class_a, class_b);
            // println!("{}, {}", result.class_a, result.class_b);
            // user_exit();
            class_a *= result.class_a;
            class_b *= result.class_b;
        }
    }

    TokenProbabilities { class_a, class_b }
}

fn naive_bayes_in_class(model: &HashMap<String, TokenProbabilities>, line: &Vec<String>) -> bool {
    let result = generate_naive_bayes_class_probability(model, line);

    result.class_a / result.class_b > 1.0
}

fn save_naive_bayes_model(
    fpath: &OsStr,
    to_save: HashMap<String, TokenProbabilities>,
) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_path(fpath)?;

    wtr.write_record(["word", "class_a", "class_b"])?;

    for item in to_save {
        wtr.write_record(&[
            item.0,
            item.1.class_a.to_string(),
            item.1.class_b.to_string(),
        ])?;
    }

    Ok(())
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
    // test_bayes_preprocess();
    // test_naive_bayes_modeling();
    test_naive_bayes_against_test();
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

//Assumes Twitter-sentiment-self-drive-DFE-Test.csv is in the crate root directory
//Scans file into a vec of LineTargets and displays the first 8 lines.
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

//Assumes Twitter-sentiment-self-drive-DFE-Test.csv is in the crate root directory
//Creates a HashMap of TokenOccurence
//Displays by iterating through the Hashmap.  Asks to terminate at each key value.
//Iterator is unordered.
fn test_bayes_preprocess() {
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
    let bayes = bayes_preprocess(&outvec, target);

    println!("{:?}", bayes.1);
    for line in bayes.0.iter() {
        println!("Key: {}", line.0);
        println!("TokenOccurence: {:?}", line.1);
        if input!("Continue? Enter 'N' to exit: ").to_uppercase() == "N" {
            std::process::exit(0);
        }
    }
}

fn test_naive_bayes_modeling() {
    let mut filepath = env::current_dir().unwrap();
    let mut savepath = env::current_dir().unwrap();
    let target = "not_relevant";
    filepath.push("Twitter-sentiment-self-drive-DFE-Training.csv");
    savepath.push("MODEL-Twitter-sentiment-self-drive-DFE-Training.csv");

    let ostringpath = filepath.into_os_string();
    if let Ok(_seepath) = ostringpath.clone().into_string() {
        //println!("{}", seepath);
    } else {
        println!("Filepath not displayable - continuing...")
    }

    let ostringsavepath = savepath.into_os_string();
    if let Ok(_seepath) = ostringsavepath.clone().into_string() {
        //println!("{}", seepath);
    } else {
        println!("Savepath not displayable - continuing...")
    }

    let outvec = parse_csv_to_linetarget(&ostringpath).unwrap();
    let bayes = bayes_preprocess(&outvec, target);
    let model = generate_naive_bayes_model(&bayes.0, bayes.1);
    save_naive_bayes_model(&ostringsavepath, model).unwrap();
}

fn test_naive_bayes_against_test() {
    let mut trainpath = env::current_dir().unwrap();
    let mut testpath = env::current_dir().unwrap();
    let target = "not_relevant";
    trainpath.push("Twitter-sentiment-self-drive-DFE-Training.csv");
    testpath.push("Twitter-sentiment-self-drive-DFE-Test.csv");

    let ostrtrainpath = trainpath.into_os_string();
    if let Ok(_seepath) = ostrtrainpath.clone().into_string() {
        //println!("{}", seepath);
    } else {
        println!("Filepath not displayable - continuing...")
    }

    let ostrtestpath = testpath.into_os_string();
    if let Ok(_seepath) = ostrtestpath.clone().into_string() {
        //println!("{}", seepath);
    } else {
        println!("Savepath not displayable - continuing...")
    }

    let outvec = parse_csv_to_linetarget(&ostrtrainpath).unwrap();
    let testvec = parse_csv_to_linetarget(&ostrtestpath).unwrap();
    let bayes = bayes_preprocess(&outvec, target);
    let model = generate_naive_bayes_model(&bayes.0, bayes.1);
    //  Finish test loop

    let mut total = 0;
    let mut correct = 0;

    //  Should work by checking first if the current item matches the target,
    //  then compares that result to the naive bayes prediction.
    for item in testvec {
        if (item.target == target) == naive_bayes_in_class(&model, &item.tokens) {
            correct += 1;
        }
        total += 1;
    }

    let percent_correct = f64::from(correct) / f64::from(total);

    println!("Correct: {}", correct);
    println!("Total: {}", total);
    println!("Percent correct: {}", percent_correct);
}
