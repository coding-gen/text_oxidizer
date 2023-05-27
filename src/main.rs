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

fn main() {
    test_open_reader();
}

//Test functions.  May or may not be unit tests.

//Assumes you have a test.txt in the current crate location
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
