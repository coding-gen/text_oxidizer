use std::{collections::HashMap, env, error::Error, ffi::OsStr};

use csv::{Reader, Writer};

pub use crate::tokenize::*;

/// Pairs a vec of tokens with a target for usage in an ML algorithm.
#[derive(Debug)]
pub struct LineTarget {
    tokens: Vec<String>,
    pub target: String,
}

/// Tracks number of tokens in two classes.
// Future revision to be extensible to n classes.
#[derive(Debug)]
pub struct TokenOccurence {
    class_a: u32,
    class_b: u32,
}

/// Track class probabilities for a token.
/// Intended to be used inside of a HashMap
// Future revision to be extensible to n classes.
#[derive(Debug)]
pub struct TokenProbabilities {
    class_a: f64,
    class_b: f64,
}

/// Captures number of words in two classes
// Future revision to be extensible to n classes.
#[derive(Debug)]
pub struct NumberWords {
    class_a: u32,
    class_b: u32,
}

/// Takes in a vec of LineTarget and the target to match against.
/// Outputs a HashMap of TokenOccurence for further processing.
/// Intended for use in Naive Bayes.
pub fn bayes_preprocess(
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

///  Takes in output of the preprocessor - a hashmap of string and occurance pairs, and the class wordcounts.
///  Outputs a model in the form of a hashmap of strings and associated class probabilities.
//  Should look at float math, and the associated i32 variables.  Some datasets may have extremely large wordcount.
//  Could potentially handle by chunking out the processing and aggregating the resulting probabilities.
pub fn generate_naive_bayes_model(
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

/// Takes in a naive bayes model in the form of a HashMap<String, TokenProbabilities> and a vec of tokens to check against
/// Returns a raw probability tuple
/// Intended to be used by naive_bayes_in_class()
fn generate_naive_bayes_class_probability(
    model: &HashMap<String, TokenProbabilities>,
    line: &Vec<String>,
) -> TokenProbabilities {
    let mut class_a = 1_f64;
    let mut class_b = 1_f64;

    for token in line {
        if let Some(result) = model.get(token) {
            class_a *= result.class_a;
            class_b *= result.class_b;
        }
    }

    TokenProbabilities { class_a, class_b }
}

/// Takes in a naive bayes model in the form of a HashMap<String, TokenProbabilities> and a vec of tokens to check against
/// Returns true if the probability indicates a match to the target built in to the model.
pub fn naive_bayes_in_class(
    model: &HashMap<String, TokenProbabilities>,
    line: &LineTarget,
) -> bool {
    let result = generate_naive_bayes_class_probability(model, &line.tokens);

    result.class_a / result.class_b > 1.0
}

pub fn naive_bayes_in_class_str(model: &HashMap<String, TokenProbabilities>, line: &str) -> bool {
    let result =
        generate_naive_bayes_class_probability(model, &tokenize_line_alphas_lowercase(line));

    result.class_a / result.class_b > 1.0
}

pub fn naive_bayes_matches_target(
    target: &str,
    model: &HashMap<String, TokenProbabilities>,
    line: &LineTarget,
) -> bool {
    (line.target == target) == naive_bayes_in_class(model, line)
}

pub fn save_naive_bayes_model(
    fpath: &OsStr,
    to_save: &HashMap<String, TokenProbabilities>,
) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_path(fpath)?;

    wtr.write_record(["word", "class_a", "class_b"])?;

    for item in to_save {
        wtr.write_record([
            item.0,
            &item.1.class_a.to_string(),
            &item.1.class_b.to_string(),
        ])?;
    }

    Ok(())
}

pub fn load_naive_bayes_model(
    fpath: &OsStr,
) -> Result<HashMap<String, TokenProbabilities>, Box<dyn Error>> {
    let mut out = HashMap::new();
    let mut reader = Reader::from_path(fpath)?;

    for result in reader.byte_records() {
        let record = result?;
        let probs = TokenProbabilities {
            class_a: String::from_utf8_lossy(record.get(1).unwrap())
                .parse::<f64>()
                .unwrap(),
            class_b: String::from_utf8_lossy(record.get(2).unwrap())
                .parse::<f64>()
                .unwrap(),
        };

        out.insert(
            String::from_utf8_lossy(record.get(1).unwrap())
                .to_string()
                .clone(),
            probs,
        );
    }

    Ok(out)
}

/// Accepts a path to a CSV file.
/// Returns a Vec of LineTargets for further processing, or any resultant errors.
pub fn parse_csv_to_linetarget(fpath: &OsStr) -> Result<Vec<LineTarget>, Box<dyn Error>> {
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

/// Ensures output of parse_csv_to_linetarget() by comparing to test.csv.
/// Expects test.csv with proper contents to be in the root directory of the crate.
/// Future update to create a temporary file with the correct contents and use this to test.
#[test]
fn test_parse_csv_to_linetarget() {
    let mut filepath = env::current_dir().unwrap();
    filepath.push("test.csv");

    let ostringpath = filepath.into_os_string();
    let outvec = parse_csv_to_linetarget(&ostringpath).unwrap();

    let line_0 = ["test", "this", "is"];
    let line_1 = ["second", "line"];

    assert_eq!(
        outvec[0].target, "a",
        "Target mismatch: {} =/= a",
        outvec[0].target,
    );
    assert_eq!(
        outvec[1].target, "b",
        "Target mismatch: {} =/= b",
        outvec[1].target
    );

    assert_eq!(
        outvec[0].tokens.len(),
        line_0.len(),
        "Mismatch in expected token length - expected {}, got {}",
        line_0.len(),
        outvec[0].tokens.len()
    );

    assert_eq!(
        outvec[1].tokens.len(),
        line_1.len(),
        "Mismatch in expected token length - expected {}, got {}",
        line_1.len(),
        outvec[1].tokens.len()
    );

    for i in 0..outvec[0].tokens.len() {
        assert_eq!(
            outvec[0].tokens[i], line_0[i],
            "Token mismatch - expected {}, got {}",
            line_0[i], outvec[0].tokens[i]
        );
    }

    for i in 0..outvec[1].tokens.len() {
        assert_eq!(
            outvec[1].tokens[i], line_1[i],
            "Token mismatch - expected {}, got {}",
            line_1[i], outvec[1].tokens[i]
        );
    }
}

/// Validates output from bayes_preprocess against test.csv
/// Expects test.csv with proper contents to be in the root directory of the crate.
/// Future update to create a temporary file with the correct contents and use this to test.
#[test]
fn test_bayes_preprocess() {
    let mut filepath = env::current_dir().unwrap();
    let target = "a";
    filepath.push("test.csv");

    let ostringpath = filepath.into_os_string();

    let outvec = parse_csv_to_linetarget(&ostringpath).unwrap();
    let (tokens, words) = bayes_preprocess(&outvec, target);

    let line_0 = ["test", "this", "is"];
    let line_1 = ["second", "line"];

    assert_eq!(
        tokens.len(),
        5,
        "Mismatch on number of lines returned from bayes_preprocess - expected 2, got {}",
        tokens.len()
    );

    assert_eq!(
        words.class_a, 3,
        "Mismatch on number of words in words.class_a - expected 3, got {}",
        words.class_a
    );
    assert_eq!(
        words.class_b, 2,
        "Mismatch on number of words in words.class_b - expected 2, got {}",
        words.class_b
    );

    for word in line_0 {
        assert_eq!(
            tokens.get(word).unwrap().class_a,
            1,
            "Expected {} in class_a, not present",
            word
        );
        assert_eq!(
            tokens.get(word).unwrap().class_b,
            0,
            "{} found in class_b, not expected",
            word
        );
    }

    for word in line_1 {
        assert_eq!(
            tokens.get(word).unwrap().class_b,
            1,
            "Expected {} in class_b, not present",
            word
        );
        assert_eq!(
            tokens.get(word).unwrap().class_a,
            0,
            "{} found in class_a, not expected",
            word
        );
    }
}

/// Validates output from bayes_preprocess against test.csv
/// Expects test.csv with proper contents to be in the root directory of the crate.
/// Future update to create a temporary file with the correct contents and use this to test.
/// Uncertain how to correctly transform this into a proper unit test
#[allow(dead_code)]
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
    save_naive_bayes_model(&ostringsavepath, &model).unwrap();
}

/// Tests trained naive bayes model against a test set.
/// Expects test.csv with proper contents to be in the root directory of the crate.
/// Future update to create a temporary file with the correct contents and use this to test.
/// Uncertain how to correctly transform this into a proper unit test
#[allow(dead_code)]
fn test_naive_bayes_against_test() {
    let mut trainpath = env::current_dir().unwrap();
    let mut testpath = env::current_dir().unwrap();
    let target = "not_relevant";
    trainpath.push("Twitter-sentiment-self-drive-DFE-Training.csv");
    testpath.push("Twitter-sentiment-self-drive-DFE-Test.csv");
    // let target = "Atheism";
    // trainpath.push("progressive-tweet-sentiment-train.csv");
    // testpath.push("progressive-tweet-sentiment-test.csv");

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
        if (item.target == target) == naive_bayes_in_class(&model, &item) {
            correct += 1;
        }
        total += 1;
        // println!(
        //     "{}, {}, {}",
        //     item.target,
        //     target,
        //     naive_bayes_in_class(&model, &item.tokens)
        // );
        // user_exit();
    }

    let percent_correct = f64::from(correct) / f64::from(total);

    println!("Correct: {}", correct);
    println!("Total: {}", total);
    println!("Percent correct: {}", percent_correct);
}
