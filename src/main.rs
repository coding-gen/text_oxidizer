mod debug_tools;
mod naive_bayes;
mod tokenize;
pub use crate::debug_tools::*;
pub use crate::naive_bayes::*;
pub use crate::tokenize::*;
use clap::arg;
use clap::Parser;

#[derive(Parser)]
struct Args {
    ///NLP tool to use
    #[arg(short, long)]
    source: String,

    ///Enable test mode - disregards filepath
    #[arg(long, default_value_t = false)]
    test: bool,

    ///Use Naive Bayes.  Cannot be used with other NLP tools.
    #[arg(long)]
    naive_bayes: bool,
}

fn main() {
    // test_naive_bayes_modeling();
    // test_naive_bayes_against_test();
    let args = Args::parse();

    println!("Argument: {}", args.source);

    if args.test {
        println!("Test is enabled");
    } else {
        println!("Test is disabled");
    }
}
