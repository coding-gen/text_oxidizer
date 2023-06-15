//! Text Oxidizer
//! Tools for Natural Language Processing in Rust
//!
//! Authors: Sawyer Norquist, Genevieve LaLonde
//! Programming in Rust 2023
//! Professor Dr. Bart Massey 2023
//! Resources:
//! https://doc.rust-lang.org/reference/

mod bpe;
mod debug_tools;
mod naive_bayes;
mod tokenize;
use std::env;
use std::error::Error;
use std::ffi::OsStr;

pub use crate::bpe::*;
pub use crate::debug_tools::*;
pub use crate::naive_bayes::*;
pub use crate::tokenize::*;
use clap::arg;
use clap::Parser;
use csv::Reader;
use csv::Writer;

/// Print a passed usage error message and exit.
/// Will panic instead if in test configuration.
fn error(err: &str) -> ! {
    eprintln!("{}", err);
    #[cfg(not(test))]
    std::process::exit(1);
    #[cfg(test)]
    panic!("error");
}

// Command line arguments
#[derive(Parser)]
struct Args {
    /// Generate Naive Bayes model.
    #[arg(long, num_args = 2, value_names = ["TARGET", "TRAINING CSV"])]
    nb_gen: Vec<String>,

    /// Generate Naive Bayes model and test it.
    #[arg(long, num_args = 3, value_names = ["TARGET", "TRAINING CSV", "TEST CSV"])]
    nb_gen_test: Vec<String>,

    /// Load Naive Bayes model and compare target string.
    #[arg(long, num_args = 2, value_names = ["SAMPLE", "MODEL CSV"])]
    nb_pred_s: Vec<String>,

    /// Load Naive Bayes model and compare CSV of strings.
    #[arg(long, num_args = 2, value_names = ["SAMPLE CSV", "MODEL CSV"])]
    nb_pred: Vec<String>,

    /// Train a BPE Tokenizer, and use it to tokenize a text file.
    #[arg(long, num_args = 2, value_names = ["SAMPLE TXT", "HYPERPARAMETER VOCAB SIZE"])]
    bpe_train: Vec<String>,

    /// Train a BPE Tokenizer, and use it to tokenize a text file.
    #[arg(long, num_args = 2, value_names = ["SAMPLE TXT", "VOCAB"])]
    bpe_tokenize: Vec<String>,
}

/// Takes in a filepath as an &OsStr
/// Returns a result containing other a Vec<String> or a resulting error
/// Used to load a sample CSV for running against an NLP ML model
fn load_csv(fpath: &OsStr) -> Result<Vec<String>, Box<dyn Error>> {
    let mut out = Vec::new();
    let mut reader = Reader::from_path(fpath)?;

    for result in reader.byte_records() {
        let record = result?;
        let target = String::from_utf8_lossy(record.get(0).unwrap()).into_owned();

        out.push(target.to_owned());
    }

    Ok(out)
}

fn main() {
    let args = Args::parse();

    if !args.nb_gen.is_empty() {
        naive_bayes_generate(args.nb_gen.get(0).unwrap(), args.nb_gen.get(1).unwrap());
    }

    if !args.nb_gen_test.is_empty() {
        naive_bayes_generate_and_test(
            args.nb_gen_test.get(0).unwrap(),
            args.nb_gen_test.get(1).unwrap(),
            args.nb_gen_test.get(2).unwrap(),
        )
    }

    if !args.nb_pred_s.is_empty() {
        naive_bayes_predict_string(
            args.nb_pred_s.get(0).unwrap(),
            args.nb_pred_s.get(1).unwrap(),
        )
    }

    if !args.nb_pred.is_empty() {
        naive_bayes_predict(args.nb_pred.get(0).unwrap(), args.nb_pred.get(1).unwrap())
    }

    if !args.bpe_train.is_empty() {
        bpe_generate(args.bpe_train.get(0).unwrap(), args.bpe_train.get(1).unwrap())
    }

    if !args.bpe_tokenize.is_empty() {
        bpe_tokenize(args.bpe_tokenize.get(0).unwrap(), args.bpe_tokenize.get(1).unwrap())
    }
}

/// Takes in a target as an &str and a filename to a training CSV as an &str
/// Assumes the CSV is in the program root folder.
/// Builds a Naive Bayes model and saves it.  Resulting CSV is saved to the program root folder.
/// Its name will be the same as the training file with 'MODEL-' appended to the front.
fn naive_bayes_generate(target: &str, training: &str) {
    let mut filepath = env::current_dir().unwrap();
    let mut savepath = env::current_dir().unwrap();
    filepath.push(training);
    savepath.push("MODEL-".to_string() + training);

    let ostringpath = filepath.into_os_string();
    let ostringsavepath = savepath.into_os_string();

    let outvec = parse_csv_to_linetarget(&ostringpath)
        .unwrap_or_else(|_| error("Cannot open or parse training CSV"));
    let bayes = bayes_preprocess(&outvec, target);
    let model = generate_naive_bayes_model(&bayes.0, bayes.1);
    save_naive_bayes_model(&ostringsavepath, &model)
        .unwrap_or_else(|_| error("Failed to save model"));
}

/// Takes in a target as an &str, a filename to a training CSV as an &str, and a filename to a test CSV as an &str
/// Assumes the CSV's are in the program root folder.
/// Builds a Naive Bayes model and saves it.  Resulting CSV is saved to the program root folder.
/// Its name will be the same as the training file with 'MODEL-' appended to the front.
/// Additionally, uses the test CSV to check the model's precision and recall.
fn naive_bayes_generate_and_test(target: &str, training: &str, test: &str) {
    let mut trainpath = env::current_dir().unwrap();
    let mut testpath = env::current_dir().unwrap();
    let mut modelpath = env::current_dir().unwrap();
    trainpath.push(training);
    testpath.push(test);
    modelpath.push("MODEL-".to_string() + training);

    let ostrtrainpath = trainpath.into_os_string();
    let ostrtestpath = testpath.into_os_string();
    let ostrmodelpath = modelpath.into_os_string();

    let outvec = parse_csv_to_linetarget(&ostrtrainpath)
        .unwrap_or_else(|_| error("Cannot open or parse training CSV"));
    let testvec = parse_csv_to_linetarget(&ostrtestpath)
        .unwrap_or_else(|_| error("Cannot open or parse test CSV"));
    let bayes = bayes_preprocess(&outvec, target);
    let model = generate_naive_bayes_model(&bayes.0, bayes.1);

    save_naive_bayes_model(&ostrmodelpath, &model)
        .unwrap_or_else(|_| error("Failed to save model"));

    let mut total: u32 = 0;
    let mut tpos: u32 = 0;
    let mut tneg: u32 = 0;
    let mut fpos: u32 = 0;
    let mut fneg: u32 = 0;

    for item in testvec {
        let result = naive_bayes_matches_target_test(target, &model, &item);
        match result {
            PredictionResult::TruePositive => tpos += 1,
            PredictionResult::TrueNegative => tneg += 1,
            PredictionResult::FalsePositive => fpos += 1,
            PredictionResult::FalseNegative => fneg += 1,
        }

        total += 1;
    }

    let percent_correct = f64::from(tpos + tneg) / f64::from(total);
    let precision = f64::from(tpos) / f64::from(tpos + fpos);
    let recall = f64::from(tpos) / f64::from(tpos + fneg);

    println!("Correct: {}", tpos + tneg);
    println!("Total: {}", total);
    println!("Percent correct: {}", percent_correct);
    println!("Precision: {}", precision);
    println!("Recall: {}", recall);
}

/// Takes in a target as an &str and a filename to a model CSV as an &str
/// Assumes the CSV is in the program root folder.
/// Compares the passed string to the model, and prints whether the prediction matches class_a in the model
fn naive_bayes_predict_string(sample: &str, model: &str) {
    let mut modelpath = env::current_dir().unwrap();
    modelpath.push(model);
    let ostrmodelpath = modelpath.into_os_string();

    let model = load_naive_bayes_model(&ostrmodelpath)
        .unwrap_or_else(|_| error("Cannot open or parse model CSV"));

    if naive_bayes_in_class_str(&model, sample) {
        println!("Sample is in-class for provided model")
    } else {
        println!("Sample is NOT in-class for provided model")
    }
}

/// Takes in a filename to a sample list as an &str and a filename to a model CSV as an &str
/// Assumes the CSV is in the program root folder.  The sample list must be a single column of sentences and include a header.
/// Compares each sentence to the model, and saves each class prediction to a CSV paired with the sample sentence.
/// Resulting file will be the same as the sample file with 'RESULT-' appended.
fn naive_bayes_predict(sample: &str, model: &str) {
    let mut modelpath = env::current_dir().unwrap();
    let mut samplepath = env::current_dir().unwrap();
    let mut outpath = env::current_dir().unwrap();
    modelpath.push(model);
    samplepath.push(sample);
    outpath.push("RESULTS-".to_string() + sample);
    let ostrmodelpath = modelpath.into_os_string();
    let ostrsamplepath = samplepath.into_os_string();
    let ostroutpath = outpath.into_os_string();

    let model = load_naive_bayes_model(&ostrmodelpath)
        .unwrap_or_else(|_| error("Cannot open or parse model CSV"));
    let samples =
        load_csv(&ostrsamplepath).unwrap_or_else(|_| error("Cannot open or parse sample CSV"));
    let mut outvec = Vec::new();

    for item in samples {
        if naive_bayes_in_class_str(&model, &item) {
            outvec.push((item, "true".to_string()));
            println!("true");
        } else {
            outvec.push((item, "false".to_string()));
        }
    }

    let mut wtr =
        Writer::from_path(&ostroutpath).unwrap_or_else(|_| error("Cannot open or parse model CSV"));

    wtr.write_record(["sample", "in-class"])
        .unwrap_or_else(|_| error("Unable to save predictions"));

    for item in outvec {
        wtr.write_record(&[item.0, item.1])
            .unwrap_or_else(|_| error("Unable to write to prediction file"));
    }

    println!(
        "Complete - file saved to: {}",
        ostroutpath.to_string_lossy()
    );
}

/// Takes in a target as an &str and a filename to a training CSV as an &str
/// Assumes the CSV is in the program root folder.
/// Builds a token vocabulary using Byte Pair Encoding and saves it.  
/// Resulting CSV is saved to the program root folder.
/// Its name will be the same as the training file with 'BPE-TOKENIZED-' appended to the front.
/// Select the hyperparameter large enough to form word roots, small enough to separate word parts like pre/suffix.
/// Can be measured, by percent of tokens in resulting vocab, which end in </w> end of word indicator.
fn bpe_generate(infile: &str, vocab_size: &str) {
    let mut filepath = env::current_dir().unwrap();
    let mut savepath = env::current_dir().unwrap();
    filepath.push(infile);
    let string_end = &infile[infile.len() - 4..];
    if string_end == ".csv"{
        savepath.push("BPE-VOCAB-".to_string() + format!("{}{}", &infile[0..infile.len() - 4], ".txt").as_str());
    } else{
        savepath.push("BPE-VOCAB-".to_string() + infile);
    }

    let ostringpath = filepath.into_os_string();
    let ostringsavepath = savepath.into_os_string();

    let outvec = parse_csv_to_lines(&ostringpath)
        .unwrap_or_else(|_| error("Cannot open or parse training CSV."));

    // TODO increase n max size, and convert so it can handle in bpe_training:
    // frequency_table.len() < n.into()
    let n = vocab_size.parse::<u8>().unwrap();
    let vocab = bpe_training(outvec, n);
    println!("Lemmatized vocab: {:?}", vocab);

    save_bpe_vocab(&ostringsavepath, &vocab).unwrap_or_else(|_| error("Failed to save vocab."));
}

fn bpe_tokenize(infile: &str, vocab_file: &str) {
    let mut filepath = env::current_dir().unwrap();
    let mut savepath = env::current_dir().unwrap();
    let mut vocabpath = env::current_dir().unwrap();

    filepath.push(infile);
    vocabpath.push(vocab_file);

    let string_end = &infile[infile.len() - 4..];
    if string_end == ".csv"{
        savepath.push("BPE-TOKENIZED-".to_string() + format!("{}{}", &infile[0..infile.len() - 4], ".txt").as_str());
    } else{
        savepath.push("BPE-TOKENIZED-".to_string() + infile);
    }

    let ostringpath = filepath.into_os_string();
    let ostringsavepath = savepath.into_os_string();
    let ostring_vocab_path = vocabpath.into_os_string();

    let outvec = parse_csv_to_lines(&ostringpath)
        .unwrap_or_else(|_| error("Cannot open or parse source CSV."));
    let vocabvec = parse_txt_to_lines(&ostring_vocab_path)
        .unwrap_or_else(|_| error("Cannot open or parse source CSV."));

    let tokenized_file = bpe_encoding(outvec, vocabvec);
    println!("Tokenized file: {:?}", tokenized_file);

    save_bpe_encoding(&ostringsavepath, &tokenized_file).unwrap_or_else(|_| error("Failed to save vocab."));
}

