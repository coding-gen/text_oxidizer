use lazy_static::lazy_static;
use regex::Regex;
use std::env;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};

//https://users.rust-lang.org/t/how-to-return-bufreader/34651/6
///Accepts a file path and returns a Result containing either a BufReader or an IO error
fn open_reader(fpath: &OsStr) -> Result<BufReader<Box<dyn Read>>, std::io::Error> {
    let fileobj = File::open(fpath)?;
    Ok(BufReader::new(Box::new(fileobj)))
}

//https://docs.rs/regex/latest/regex/
//Using lazy_static as recommended by regex crate docs
fn tokenize(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();

    println!("tokenize: {}", line);

    lazy_static! {
        static ref REGTOKEN: Regex = Regex::new(r"\S+").unwrap();
    }

    // println!("{:?}", REGTOKEN);
    for group in REGTOKEN.captures_iter(line) {
        println!("{:?}", group);
        for c in group.iter().skip(1) {
            tokens.push(c.unwrap().as_str().to_string())
        }
    }
    tokens
}

fn main() {
    test_open_reader();
    test_tokenize();
}

//Test functions.  May or may not be unit tests.

//Assumes you have a test.txt in the current crate location
//Reads test.txt line by line, printing each line
fn test_open_reader() {
    let mut filepath = env::current_dir().unwrap();
    filepath.push("test.txt");
    let ostringpath = filepath.into_os_string();
    if let Ok(seepath) = ostringpath.clone().into_string() {
        println!("{}", seepath);
    } else {
        println!("Filepath not displayable - continuing...")
    }
    let nreader = open_reader(&ostringpath);

    if let Ok(linereader) = nreader {
        for line in linereader.lines() {
            println!("{}", line.unwrap())
        }
    } else {
        println!("Invalid filepath")
    }
}

fn test_tokenize() {
    let mut filepath = env::current_dir().unwrap();
    let mut outvec: Vec<String> = Vec::new();
    filepath.push("test.txt");
    let ostringpath = filepath.into_os_string();
    if let Ok(seepath) = ostringpath.clone().into_string() {
        println!("{}", seepath);
    } else {
        println!("Filepath not displayable - continuing...")
    }
    let nreader = open_reader(&ostringpath);

    if let Ok(linereader) = nreader {
        for line in linereader.lines() {
            outvec.append(&mut tokenize(&line.unwrap()))
        }
    } else {
        println!("Invalid filepath")
    }

    for s in outvec.iter() {
        println!("{}", s);
    }
}
