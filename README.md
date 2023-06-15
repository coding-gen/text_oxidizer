# text_oxidizer

Final Project in course: Programming in Rust
Natural Language Processing toolkit in Rust.

## Authors:

Sawyer Norquist, Genevieve LaLonde

### White Space Tokenizer

The Tokenizer crate is intended to be used in conjunction with other NLP activities. It consists of a handful of functions for parsing an input and returning a vector of strings representing the tokenized stream.

### Byte Pair Encoder Tokenizer

The Byte Pair Encoder Tokenizer crate can be used to generate a token vocabulary from a corpus, as well as to tokenize a corpus from that generated vocab. BPE is a more precise form of tokenization that is better able to represent word roots and parts of speech like prefixes and suffixes. For more information see this [medium article on BPE](https://towardsdatascience.com/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0).

#### To Generate the Vocabulary

```
--bpe-train <SAMPLE CSV> <HYPERPARAMETER VOCAB SIZE>
	Determines statistical likelihood of letter combinations in the provided CSV file. Generates a vocabulary of the specified size. Set the size large enough to form word roots, but small enough to separate word parts like pre/suffix. Some early language models used a vocab size of roughly 50,000 tokens. Expects a two column csv input corpus, where the sequences to encode are in the second column. The vocab is generated in the same directory, with `BPE-VOCAB-` prepended to it.
```

#### Tokenize a Corpus from a Vocabulary

```
--bpe-tokenize <SAMPLE CSV> <VOCAB TXT>
	Uses the provided vocab text file to tokenize each sequence of the sample file. Expects a two column csv input corpus, where the sequences to encode are in the second column. The tokenized corpus is provided as a file in the same directory named for the corpus, prepended with `BPE-TOKENIZED-`.
```

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

```
--nb-gen <TARGET> <TRAINING CSV>
Generates and saves a Naive Bayes model from the Training CSV against the target string. The output file will be the same name as the training CSV with 'MODEL-' appended to the front.

--nb-gen-test <TARGET> <TRAINING CSV> <TEST CSV>
Effectively the same as --nb-gen. The added parameter allows for a test CSV to be loaded and run against the trained model. In addition to creating and saving the model, prints correct predictions, total predictions, percent correct, precision, and recall.

--nb-pred-s <SAMPLE> <MODEL CSV>
Loads the model CSV and compares the sample as a string. Prints if the sample is predicted within the model class or not.

--nb-pred <SAMPLE CSV> <MODEL CSV>
Used to batch process samples against the model. Saves a CSV by the same name as the sample file with 'RESULTS-' appended to the front. The first column will be the original sample strings, and the second column will indicate 'true' if the associated sample is predicted to be within class or 'false' if not.
```

## Testing

During development, testing was performed while coding using the test.txt and test.csv files, as well as the Twitter sentiment datasets on both progressive viewpoints, as well as about self driving cars. Additionally tests can be located at the bottom of each crate. To run a test:

```
cargo test
```


## Limitations and Future Directions

Performance could be improved by switching to more of an object oriented paradigm to prevent passing large parameters back and forth. In Naive Bayes, smoothing could be implemented to prevent predictions from approaching 0 on unseen values. For example see this [Medium article on Laplacian Smoothing in Naive Bayes](https://towardsdatascience.com/laplace-smoothing-in-naïve-bayes-algorithm-9c237a8bdece). Additionally in the BPE function for selecting tokens to merge, the candidate table could be persisted and updated between runs instead of recreated and recomputed at each merge. Additionally the current vocab limit for the BPE is 255 due to the datatype selected(`u8`).

## License

Limited license available at: [LICENSE](https://github.com/coding-gen/text_oxidizer/blob/main/LICENSE).

## Demo

Demo video available at: [Text Oxidizer Demo](https://youtu.be/R3Yf2Qyh_1o).
