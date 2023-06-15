# text_oxidizer

Final Project in course: Programming in Rust
Natural Language Processing toolkit in Rust.

## Authors:

Sawyer Norquist, Genevieve LaLonde

## Project Description

## Usage Statement

### Tokenizer

The Tokenizer crate is intended to be used in conjunction with other NLP activities. It consists of a handful of functions for parsing an input and returning a vector of strings representing the tokenized stream.

### Naive Bayes

The Naive Bayes modeling can be accessed two ways. One, directly via using the crate as a library. Two, by taking advantage of the command line interface.

#### As a library

Examples of how to use the methods can be found in main.rs, but are principly broken out into four steps:

1. Convert the input into a string of tokens via a tokenizer method. tokenize_line_alphas_lowercase() is recommended for this purpose. The result must be a vector of LineTokens, each composed of a token stream and the class target. Alternatively, parse_csv_to_linetarget() is a build-in method to convert a CSV into the correct data structure.
2. Call bayes_preprocess() on the vector of LineTokens, including the target class to be processed. This will return a tuple. The first element is a HashMap of tokens and TokenOccurence to indicate the number of token occurences for each class. The second element is a NumberWords struct to indicate the total number of tokens in each class.
3. Call generate_naive_bayes_model() on the tuple passed out of bayes_preprocess(). This will generate the statistical information needed to predict a target token stream.
4. Compare a sample to the model. Two methods exist to do this. First, naive_bayes_in_class_str() will accept a string and a Naive Bayes model, and return true if the string is likely to fall within the model's target class. Second, if instead your implementation has a LineTarget directly, naive_bayes_in_class() can be used to process it instead with the same results.

Additionally, there are a few utility functions. save_naive_bayes_model() and load_naive_bayes_model() allow the trained model to be saved and loaded to disc in the form of an SSD. naive_bayes_matches_target() and naive_bayes_matches_target_test() can be used to validate the trained model. The former simply returns true if the target matches the model prediction from the sample. The latter returns an enum indicating the nature of the match - true and false positives, as well as true and false negatives.

#### On the command line

This anticipates all CSV files to be in the root directory of the program. Additionally, each CSV must include headers, though they are discarded. Sample CSV's must be a single column of strings, while the training and test CSV's must have the target classes in column 1 and the strings to compare in column two.

There are four arguments set up to run Naive Bayes:

--nb-gen <TARGET> <TRAINING CSV>
Generates and saves a Naive Bayes model from the Training CSV against the target string. The output file will be the same name as the training CSV with 'MODEL-' appended to the front.

--nb-gen-test <TARGET> <TRAINING CSV> <TEST CSV>
Effectively the same as --nb-gen. The added parameter allows for a test CSV to be loaded and run against the trained model. In addition to creating and saving the model, prints correct predictions, total predictions, percent correct, precision, and recall.

--nb-pred-s <SAMPLE> <MODEL CSV>
Loads the model CSV and compares the sample as a string. Prints if the sample is predicted within the model class or not.

--nb-pred <SAMPLE CSV> <MODEL CSV>
Used to batch process samples against the model. Saves a CSV by the same name as the sample file with 'RESULTS-' appended to the front. The first column will be the original sample strings, and the second column will indicate 'true' if the associated sample is predicted to be within class or 'false' if not.

## Testing

## Illustration of Operation

## Limitations and Future Directions

## License

Limited license available at: [LICENSE](https://github.com/coding-gen/text_oxidizer/blob/main/LICENSE).

## Demo

To be published.

---

Your name(s)

Your project name

A description of your project (1-2 paragraphs, enough so that people know what it is, what it does, how it works
What is the project? What does it do — what is its intended function?
How do folks build and run the project, in plenty of detail?
How was testing done to make sure the project works?

An example illustrating the operation of your code

What worked? What didn't? How satisfied are you with the result? What would you like to improve in the future?

License information if any, including a pointer to the LICENSE file.
