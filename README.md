# number-general
A generic Rust number type with support for basic math operations, (de)serialization, and casting with
[safecast](http://github.com/haydnv/safecast).

Example usage:
```rust
use number_general::{Int, Number};
use safecast::CastFrom;

let sequence: Vec<Number> = serde_json::from_str("[true, 2, 3.5, -4, [1.0, -0.5]]").unwrap();
let actual = sequence.into_iter().product();

assert_eq!(actual, Number::from(num::Complex::<f64>::new(-28., 14.)));
assert_eq!(Int::cast_from(actual), Int::from(-28));
```