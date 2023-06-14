mod debug_tools;
mod naive_bayes;
mod tokenize;
use std::env;
use std::error::Error;
use std::ffi::OsStr;

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

#[derive(Parser)]
struct Args {
    /// Generate Naive Bayes model.  Requires <TARGET> <TRAINING CSV PATH>
    /// Will save file based on the training CSV filename
    #[arg(long, num_args = 2)]
    nb_gen: Vec<String>,

    /// Generate Naive Bayes model and test it.  Requires <TARGET> <TRAINING CSV PATH> <TEST CSV PATH>
    /// Will show test results and save file based on the training CSV filename
    #[arg(long, num_args = 3)]
    nb_gen_test: Vec<String>,

    /// Load Naive Baye model and compare target string.  Requires <SAMPLE STRING> <MODEL CSV>
    /// Will return True if the sample string is a statistical fit for the class the model was trained against.
    /// Does not return the description of the class.
    #[arg(long, num_args = 2)]
    nb_pred_s: Vec<String>,

    /// Load Naive Baye model and compare CSV of strings.  Requires <SAMPLE CSV> <MODEL CSV>
    /// Will return CSV of strings with indicators if the string is a statistical fit for the class the model was trained against.
    /// Does not return the description of the class.
    #[arg(long, num_args = 2)]
    nb_pred: Vec<String>,
}

pub fn load_csv(fpath: &OsStr) -> Result<Vec<String>, Box<dyn Error>> {
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
}

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

    //  Should work by checking first if the current item matches the target,
    //  then compares that result to the naive bayes prediction.
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
