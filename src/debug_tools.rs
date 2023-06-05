// Source code from:
// https://stackoverflow.com/questions/21747136/how-do-i-print-in-rust-the-type-of-a-variable

/*
// Add to file where to debug:

mod debug_tools;
pub use crate::debug_tools::*;

// usage examples:
print_type_of(&s); // &str
print_type_of(&i); // i32
*/

pub fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}
