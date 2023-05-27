use std::ffi::OsString;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};

//https://users.rust-lang.org/t/how-to-return-bufreader/34651/6
fn open_reader(fpath: &OsString) -> BufReader<Box<dyn Read>> {
    let fileobj = Box::new(File::open(fpath).unwrap());
    BufReader::new(fileobj)
}

fn main() {
    println!("Hello, world!");
    let filepath = "C:\\RustRepo\\text_oxidizer\\test.txt".to_string();
    println!("{}", filepath);
    let nreader = open_reader(&OsString::from(filepath));
    for line in nreader.lines() {
        println!("{}", line.unwrap())
    }
}
