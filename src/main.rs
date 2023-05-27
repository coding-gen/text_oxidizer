use std::ffi::OsString;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};

//https://users.rust-lang.org/t/how-to-return-bufreader/34651/6
fn open_reader(fpath: &OsString) -> Result<BufReader<Box<dyn Read>>, std::io::Error> {
    let fileobj = File::open(fpath)?;
    Ok(BufReader::new(Box::new(fileobj)))
}

fn main() {
    println!("Hello, world!");
    let filepath = "C:\\RustRepo\\text_oxidizer\\test.txt".to_string();
    println!("{}", filepath);
    let nreader = open_reader(&OsString::from(filepath));

    if let Ok(linereader) = nreader {
        for line in linereader.lines() {
            println!("{}", line.unwrap())
        }
    } else {
        println!("Invalid filepath")
    }
}
